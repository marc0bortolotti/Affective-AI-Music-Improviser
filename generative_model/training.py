import glob
import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
import matplotlib.pyplot as plt
import yaml
import re
from tokenization import PrettyMidiTokenizer, BCI_TOKENS, SILENCE_TOKEN
from model import TCN
from tokenization import Dictionary
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# MODEL PARAMETERS
EPOCHS = 500 # 500
LEARNING_RATE = 0.002 # 4
BATCH_SIZE = 32 # 16
TRAIN_MODEL = True
FEEDBACK = False
EMPHASIZE_EEG = False
EARLY_STOP = True

DIRECTORY_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(DIRECTORY_PATH, 'dataset')


def tokenize_midi_files():

    input_filenames = sorted(glob.glob(os.path.join(DATASET_PATH, 'rythms/*.mid')))
    output_filenames = sorted(glob.glob(os.path.join(DATASET_PATH, 'melodies/*.mid')))

    INPUT_TOK = PrettyMidiTokenizer()
    OUTPUT_TOK = PrettyMidiTokenizer()

    for i, (in_file, out_file) in enumerate(zip(input_filenames, output_filenames)):

        in_file_name = os.path.basename(in_file)
        out_file_name = os.path.basename(out_file)
        print(f'\n{i + 1}: {in_file_name} -> {out_file_name}')

        if 'RELAXED' in in_file_name:
            emotion_token = BCI_TOKENS[0]
        elif 'CONCENTRATED' in in_file_name:
            emotion_token = BCI_TOKENS[1]
        else:
            raise Exception('Emotion not found in file name. Please add the emotion to the file name.')

        print(f'Emotion token: {emotion_token}')

        in_seq, in_df = INPUT_TOK.midi_to_tokens(in_file, update_sequences= True, update_vocab=True, emotion_token = emotion_token, instrument='drum')
        out_seq, out_df = OUTPUT_TOK.midi_to_tokens(out_file, update_sequences= True, update_vocab=True)

        if len(INPUT_TOK.sequences) != len(OUTPUT_TOK.sequences):
            min_len = min(len(INPUT_TOK.sequences), len(OUTPUT_TOK.sequences))
            INPUT_TOK.sequences = INPUT_TOK.sequences[:min_len]
            OUTPUT_TOK.sequences = OUTPUT_TOK.sequences[:min_len]

    print(f'\nNumber of input sequences: {len(INPUT_TOK.sequences)}')
    print(f'Input sequence length: {len(INPUT_TOK.sequences[0])}')
    print(f'Input vocabulars size: {len(INPUT_TOK.VOCAB)}')
    print(f'\nNumber of output sequences: {len(OUTPUT_TOK.sequences)}')
    print(f'Output sequence length: {len(OUTPUT_TOK.sequences[0])}')
    print(f'Output vocabulars size: {len(OUTPUT_TOK.VOCAB)}')

    return INPUT_TOK, OUTPUT_TOK


def update_sequences(INPUT_TOK, OUTPUT_TOK):
    
    for tokenizer in [INPUT_TOK, OUTPUT_TOK]:

        token_frequency_threshold = 0.001 / 100
        original_vocab = tokenizer.VOCAB
        tokenizer.VOCAB.compute_weights()

        # Remove tokens that appear less than # times in the dataset
        for idx, weigth in enumerate(original_vocab.weights):
            if weigth < token_frequency_threshold:
                original_vocab.counter[idx] = 0

        # Create a new vocab with the updated tokens
        updated_vocab = Dictionary()
        updated_vocab.add_word(SILENCE_TOKEN)
        updated_vocab.add_word(BCI_TOKENS[0])
        updated_vocab.add_word(BCI_TOKENS[1])
        for word in original_vocab.word2idx.keys():
            if original_vocab.counter[original_vocab.word2idx[word]] > 0:
                updated_vocab.add_word(word)

        # Verify that the sequences were updated
        seq = tokenizer.sequences[0][:20].copy()
        seq = [original_vocab.idx2word[tok] for tok in seq]
        print(f'Initial sequence: {seq}')

        # Update the sequences with the new vocab
        for seq in tokenizer.sequences:
            for i, tok in enumerate(seq):
                if original_vocab.counter[tok] == 0 and original_vocab.idx2word[tok] not in BCI_TOKENS.values():
                    seq[i] = updated_vocab.word2idx[SILENCE_TOKEN]
                    updated_vocab.add_word(SILENCE_TOKEN)
                else:
                    word = original_vocab.idx2word[tok]
                    seq[i] = updated_vocab.word2idx[word]
                    updated_vocab.add_word(word)
        
        tokenizer.VOCAB = updated_vocab
        tokenizer.VOCAB.compute_weights()

        # Verify that the sequences were updated
        seq = tokenizer.sequences[0][:20].copy()
        seq = [tokenizer.VOCAB.idx2word[tok] for tok in seq]
        print(f'Updated sequence: {seq}')

        # Verify that the vocab was updated
        print(f'Inintial number of tokens: {len(original_vocab)}')
        print(f'Final number of tokens: {len(tokenizer.VOCAB)}\n')

    return INPUT_TOK, OUTPUT_TOK
    

def create_dataset(split = [0.7, 0.2, 0.1]):
    dataset = TensorDataset(torch.LongTensor(INPUT_TOK.sequences).to(device),
                            torch.LongTensor(OUTPUT_TOK.sequences).to(device))

    # Split the dataset into training, evaluation and test sets
    train_set, eval_set, test_set = random_split(dataset, split)

    return train_set, eval_set, test_set

def data_augmentation_shift(dataset, shifts):
    '''
    Shifts the sequences by a number of ticks to create new sequences.
    '''
    augmented_input_sequences = []
    output_sequences = []

    for ticks in shifts:
        for input_sequence, ouput_sequence in dataset:
            input_sequence = input_sequence.cpu().numpy().copy()

            # remove the first token since it is the emotion token
            emotion_token = input_sequence[0]
            input_sequence = input_sequence[1:]

            # shift the sequence
            new_input_sequence = np.roll(input_sequence, ticks)

            # add the emotion token back to the sequence
            new_input_sequence = np.concatenate(([emotion_token], new_input_sequence))

            # add the new sequence to the augmented sequences
            augmented_input_sequences.append(new_input_sequence)
            output_sequences.append(ouput_sequence.cpu().numpy().copy())
    
    augmented_dataset = TensorDataset(torch.LongTensor(augmented_input_sequences).to(device), 
                                      torch.LongTensor(output_sequences).to(device))
    
    # Concatenate the original and the augmented dataset
    concatenated_dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset])

    return concatenated_dataset


def data_augmentation_transposition(dataset, transpositions, probability=0.5):
    '''
    Transpose the sequences by a number of semitones to create new sequences.

    Parameters:
    - transpositions: a list of integers representing the number of semitones to transpose the sequences.

    NB: The transposition is done by adding the number of semitones to the pitch of each note in the sequence.
    '''

    input_sequences = []
    augmented_output_sequences = []

    for transposition in transpositions:
        for input_sequence, ouput_sequence in dataset:

            input_sequence = input_sequence.cpu().numpy().copy()
            new_ouput_sequence = ouput_sequence.cpu().numpy().copy()

            for i in range(len(new_ouput_sequence)):

                token = ouput_sequence[i]
                word = OUTPUT_TOK.VOCAB.idx2word[token]

                # check if the token is a note
                if word != SILENCE_TOKEN and word != BCI_TOKENS['relaxed'] and word != BCI_TOKENS['concentrated']:

                    # extract all the pitches from the token 
                    pitches = re.findall(r'\d+', word) # NB: pitches is a string list

                    # transpose pitch in the token with a probability
                    for pitch in pitches:
                        if np.random.rand() < probability:
                            new_pitch = str(int(pitch) + transposition)
                            word = word.replace(pitch, new_pitch)

                    # add the new token to the vocabulary
                    OUTPUT_TOK.VOCAB.add_word(word) 

                    # update the sequence with the new token
                    new_ouput_sequence[i] = OUTPUT_TOK.VOCAB.word2idx[word]
            
            # update sequence with the new tokens
            input_sequences.append(input_sequence)
            augmented_output_sequences.append(new_ouput_sequence)

    augmented_dataset = TensorDataset(torch.LongTensor(input_sequences).to(device), 
                                      torch.LongTensor(augmented_output_sequences).to(device))
    
    # Concatenate the original and the augmented dataset
    concatenated_dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset])

    return concatenated_dataset

def initialize_dataset(train_set, eval_set, test_set):

    # Create the dataloaders
    train_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=BATCH_SIZE)

    eval_sampler = RandomSampler(eval_set)
    eval_dataloader = DataLoader(eval_set, sampler=eval_sampler, batch_size=BATCH_SIZE)

    test_sampler = RandomSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, eval_dataloader, test_dataloader


def initialize_model():

    '''
    IMPORTANT:
    to cover all the sequence of tokens k * d must be >= hidden units (see the paper)
    k = kernel_size
    d = dilation = 2 ^ (n_levels - 1)
    '''

    global SEED, INPUT_SIZE, EMBEDDING_SIZE, LEVELS, HIDDEN_UNITS, NUM_CHANNELS, OUTPUT_SIZE, LOSS_WEIGTHS

    SEED = 1111
    torch.manual_seed(SEED)

    OUTPUT_SIZE = len(OUTPUT_TOK.VOCAB)

    if FEEDBACK:
        INPUT_SIZE = len(INPUT_TOK.VOCAB) + OUTPUT_SIZE
        LEVELS = 8
        HIDDEN_UNITS = INPUT_TOK.SEQ_LENGTH * 2 # 192 * 2 = 384
    else:
        INPUT_SIZE = len(INPUT_TOK.VOCAB)
        LEVELS = 7
        HIDDEN_UNITS = INPUT_TOK.SEQ_LENGTH # 192

    EMBEDDING_SIZE = 20 # size of word embeddings -> Embedding() is used to encode input token into [192, 20] real value vectors (see model.py)
    NUM_CHANNELS = [HIDDEN_UNITS] * (LEVELS - 1) + [EMBEDDING_SIZE] # [192, 192, 192, 192, 192, 192, 20]

    # balance the loss function by assigning a weight to each token related to its frequency
    LOSS_WEIGTHS = torch.ones([OUTPUT_SIZE], dtype=torch.float, device = device)
    for i, weigth in enumerate(OUTPUT_TOK.VOCAB.weights):
        LOSS_WEIGTHS[i] = 1 - weigth

    # create the model
    model = TCN(input_size = INPUT_SIZE,
                embedding_size = EMBEDDING_SIZE,
                output_size = OUTPUT_SIZE,
                num_channels = NUM_CHANNELS,
                emphasize_eeg = EMPHASIZE_EEG,
                dropout = 0.45,
                emb_dropout = 0.25,
                kernel_size = 3,
                tied_weights = False) # tie encoder and decoder weights (legare)

    model.to(device)

    # May use adaptive softmax to speed up training
    criterion = nn.CrossEntropyLoss(weight = LOSS_WEIGTHS)
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer

def save_parameters():

    # plot the losses over the epochs

    plt.plot(train_losses, label='train')
    plt.plot(eval_losses, label='eval')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, 'losses.png'))
    plt.clf()

    # save the vocabularies
    INPUT_TOK.VOCAB.save(os.path.join(RESULTS_PATH, 'input_vocab.txt'))
    OUTPUT_TOK.VOCAB.save(os.path.join(RESULTS_PATH, 'output_vocab.txt'))

     # save the model hyperparameters in a file txt
    with open(os.path.join(RESULTS_PATH, 'model_hyperparameters.txt'), 'w') as f:

        f.write(f'DATE: {time.strftime("%Y%m%d-%H%M%S")}\n\n')

        f.write(f'-----------------DATASET------------------\n')
        f.write(f'DATASET_PATH: {DATASET_PATH}\n')
        f.write(f'TRAIN_SET_SIZE: {len(train_set)}\n')
        f.write(f'EVAL_SET_SIZE: {len(eval_set)}\n')
        f.write(f'TEST_SET_SIZE: {len(test_set)}\n\n')


        f.write(f'----------OPTIMIZATION PARAMETERS----------\n')
        f.write(f'GRADIENT_CLIP: {GRADIENT_CLIP}\n')
        f.write(f'FEEDBACK: {FEEDBACK}\n')
        f.write(f'EARLY STOPPING: {EARLY_STOP}\n')
        f.write(f'EMPHASIZE_EEG: {EMPHASIZE_EEG}\n')
        f.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
        f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
        f.write(f'EPOCHS: {EPOCHS}\n\n')


        f.write(f'------------MODEL PARAMETERS--------------\n')
        f.write(f'SEED: {SEED}\n')
        f.write(f'INPUT_SIZE: {INPUT_SIZE}\n')
        f.write(f'EMBEDDING_SIZE: {EMBEDDING_SIZE}\n')
        f.write(f'LEVELS: {LEVELS}\n')
        f.write(f'HIDDEN_UNITS: {HIDDEN_UNITS}\n')
        f.write(f'NUM_CHANNELS: {NUM_CHANNELS}\n')
        f.write(f'OUTPUT_SIZE: {OUTPUT_SIZE}\n')
        f.write(f'LOSS_WEIGTHS: {LOSS_WEIGTHS}\n\n')



        f.write(f'-------------------RESULTS----------------\n')
        f.write(f'TRAIN_LOSSES: {best_train_loss}\n')
        f.write(f'BEST_EVAL_LOSS: {best_eval_loss}\n')
        f.write(f'TEST_LOSS: {test_loss}\n')
        f.write(f'BEST_TRAIN_ACCURACY: {best_train_accuracy}\n')
        f.write(f'BEST_EVAL_ACCURACY: {best_eval_accuracy}\n')
        f.write(f'TEST_ACCURACY: {test_accuracy}\n')
        f.write(f'BEST_MODEL_EPOCH: {best_model_epoch}\n')

    data = {
        'DATE': time.strftime("%Y%m%d-%H%M%S"),
        'INPUT_SIZE': INPUT_SIZE,
        'EMBEDDING_SIZE': EMBEDDING_SIZE,
        'NUM_CHANNELS': NUM_CHANNELS,
        'OUTPUT_SIZE': OUTPUT_SIZE,
        'KERNEL_SIZE': 3
    }

    path = os.path.join(RESULTS_PATH, 'config.yaml')
    with open(path, 'w') as file:
        yaml.safe_dump(data, file)


def epoch_step(dataloader, mode):

    if FEEDBACK:
        prev_output = torch.zeros([BATCH_SIZE, INPUT_TOK.SEQ_LENGTH], dtype=torch.long, device=device)

    if mode == 'train':
        model.train()
    else:
        model.eval() # disable dropout

    total_loss = 0
    accuracy.reset()

    # iterate over the training data
    for batch_idx, (data, targets) in enumerate(dataloader):

        batch_idx += 1

        # mask the last bar of the input data
        batch_size = data.size(0)
        data_masked = torch.cat((data[:, :BAR_LENGTH*3], torch.ones([batch_size, BAR_LENGTH], dtype=torch.long, device = device)), dim = 1)

        if FEEDBACK:
            input = torch.cat((data_masked, prev_output[:batch_size, :]), dim = 1)
        else:
            input = data_masked

        # reset model gradients to zero
        optimizer.zero_grad()

        # make the prediction
        output = model(input)[:, :INPUT_TOK.SEQ_LENGTH]
        prev_output = torch.argmax(output, 2)# batch, seq_len (hidden units), vocab_size

        # flatten the output sequence
        # NB: the size -1 is inferred from other dimensions
        # NB: contiguous() is used to make sure the tensor is stored in a contiguous chunk of memory, necessary for view() to work

        final_target = targets.contiguous().view(-1)
        final_output = output.contiguous().view(-1, OUTPUT_SIZE)

        accuracy.update(final_output, final_target)

        # calculate the loss
        loss = criterion(final_output, final_target)

        if mode == 'train':
            # calculate the gradients
            loss.backward()

            # clip the gradients to avoid exploding gradients
            if GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # update the weights
            optimizer.step()

        total_loss += loss.data.item()

    return total_loss / len(dataloader), accuracy.compute()


def train(results_path = None):

    global RESULTS_PATH, MODEL_PATH
    global best_eval_loss, best_train_loss, best_model_epoch, train_losses, eval_losses
    global best_train_accuracy, best_eval_accuracy

    if results_path is None:
        RESULTS_PATH = os.path.join('results', time.strftime("%Y%m%d_%H%M%S"))
    else:
        RESULTS_PATH = results_path
    
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    MODEL_PATH = os.path.join(RESULTS_PATH, 'model_state_dict.pth')

    best_eval_loss = 1e8
    best_train_loss = 1e8
    best_eval_accuracy = 0
    best_train_accuracy = 0
    best_model_epoch = 0
    eval_losses = []
    train_losses = []
    lr = LEARNING_RATE

    writer = SummaryWriter()

    for epoch in range(1, EPOCHS+1):

        start_time = time.time()

        train_loss, train_accuracy = epoch_step(train_dataloader, 'train')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        eval_loss, eval_accuracy = epoch_step(eval_dataloader, 'eval')
        writer.add_scalar('Loss/eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/eval', eval_accuracy, epoch)

        # Save the model if the validation loss is the best we've seen so far.
        if eval_loss < best_eval_loss:
            # torch.save(model.state_dict(), MODEL_PATH)
            best_eval_loss = eval_loss
            best_model_epoch = epoch

        if train_loss < best_train_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_train_loss = train_loss

        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
        
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        # Anneal the learning rate if the validation loss plateaus
        if epoch > 5 and eval_loss >= max(train_losses[-5:]):
            lr = lr / 2.
            if lr < 0.000001:
                lr = LEARNING_RATE
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        eval_losses.append(eval_loss)
        train_losses.append(train_loss)

        # Early stopping
        if EARLY_STOP:
          if epoch > 15:
              if min(train_losses[-15:]) > best_train_loss:
                  break

        # print the loss and the progress
        elapsed = time.time() - start_time
        print('| epoch {:3d}/{:3d} | lr {:02.5f} | ms/epoch {:5.5f} | train_acc {:5.2f} | train_loss {:5.2f}' \
                .format(epoch, EPOCHS, lr, elapsed * 1000, train_accuracy, train_loss))


    print('\n\n TRAINING FINISHED:\n\n\tBest Loss: {:5.2f}\tBest Model saved at epoch: {:3d} \n\n' \
            .format(best_eval_loss, best_model_epoch))


    # test the model
    global test_loss, test_accuracy
    test_loss, test_accuracy = epoch_step(test_dataloader, 'eval')
    print(f'\n\nTEST LOSS: {test_loss}')
    print(f'TEST ACCURACY: {test_accuracy}')

    save_parameters()

    writer.flush()
    writer.close()
    

if __name__ == '__main__':

    # tokenize the midi files
    global INPUT_TOK, OUTPUT_TOK
    INPUT_TOK, OUTPUT_TOK = tokenize_midi_files()

    # update the sequences
    INPUT_TOK, OUTPUT_TOK = update_sequences(INPUT_TOK, OUTPUT_TOK)

    # create the dataset
    train_set, eval_set, test_set = create_dataset()
    print(f'Train set size: {len(train_set)}')
    print(f'Evaluation set size: {len(eval_set)}')
    print(f'Test set size: {len(test_set)}')

    # augment the dataset
    train_set_augmented = data_augmentation_shift(train_set, [-2, -1, 1, 2])
    print(f'Training set size after augmentation: {len(train_set_augmented)}')

    # initialize the dataloaders
    print(f'Initializing the dataloaders...')
    train_dataloader, eval_dataloader, test_dataloader = initialize_dataset(train_set_augmented, eval_set, test_set)
    
    # initialize the model
    print(f'Initializing the model...')
    global model, criterion, optimizer
    model, criterion, optimizer = initialize_model()

    # train the model
    print(f'Training the model...')
    train('results/model_test')
