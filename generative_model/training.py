import glob
import os
import time
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
import matplotlib.pyplot as plt
import yaml
from tokenization import PrettyMidiTokenizer, BCI_TOKENS
from architectures.transformer import TransformerModel, generate_square_subsequent_mask
from architectures.musicTransformer import MusicTransformer
from architectures.tcn import TCN   
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from data_augmentation import data_augmentation_shift
from losses import CrossEntropyWithPenaltyLoss
import random

DIRECTORY_PATH = os.path.dirname(__file__)

MODEL_NAME = 'MT'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\n', device)

COMBINE_IN_OUT_TOKENS = False # combine the input and the output tokens in the same sequence
FROM_MELODY_TO_RHYTHM = True # train the model to generate rythms from melodies

GEN_TYPE = 'rhythm' if FROM_MELODY_TO_RHYTHM else 'melody'
TOK_TYPE = 'uniqueTokens' if COMBINE_IN_OUT_TOKENS else 'separateTokens'
RESULTS_PATH = os.path.join(DIRECTORY_PATH, f'runs/{MODEL_NAME}_{GEN_TYPE}_{TOK_TYPE}_0')
DATASET_PATH = os.path.join(DIRECTORY_PATH, 'dataset')

SEED = 1111
torch.manual_seed(SEED)

EPOCHS = 1000 
LEARNING_RATE = 0.0001 # 0.002
BATCH_SIZE = 64 # 64

ARCHITECTURES = {'T': TransformerModel, 'TCN' : TCN, 'MT': MusicTransformer}
try:
    MODEL = ARCHITECTURES[MODEL_NAME]
except:
    raise Exception('Model not found, check the model name')

USE_EEG = True # use the EEG data to condition the model
FEEDBACK = False # use the feedback mechanism in the model
EMPHASIZE_EEG = False # emphasize the EEG data in the model (increase weights)
DATA_AUGMENTATION = False # augment the dataset by shifting the sequences
LR_SCHEDULER = True # use a learning rate scheduler to reduce the learning rate when the loss plateaus

TICKS_PER_BEAT = 4 if FROM_MELODY_TO_RHYTHM else 4
N_TOKENS = TICKS_PER_BEAT # number of tokens to be predicted at ea@ch forward pass (only for the transformer model)

EMBEDDING_SIZE = 128
TOKENS_FREQUENCY_THRESHOLD = None # remove tokens that appear less than # times in the dataset
SILENCE_TOKEN_WEIGHT = 0.01 # weight of the silence token in the loss function
CROSS_ENTROPY_WEIGHT = 1.0  # weight of the cross entropy loss in the total loss
PENALTY_WEIGHT = 1.0 # weight of the penalty term in the total loss (number of predictions equal to class SILENCE)

GRADIENT_CLIP = 0.35 # clip the gradients to avoid exploding gradients
DATASET_SPLIT = [0.8, 0.1, 0.1] # split the dataset into training, evaluation and test sets
EARLY_STOP_EPOCHS = 15  # stop the training if the loss does not improve for # epochs
LR_PATIENCE = 10   # reduce the learning rate if the loss does not improve for # epochs

# create a unique results path
idx = 1
while os.path.exists(RESULTS_PATH):
    RESULTS_PATH = RESULTS_PATH[:-1] + str(idx)
    idx += 1
os.makedirs(RESULTS_PATH)

def initialize_model(INPUT_TOK, OUTPUT_TOK):

    if MODEL == TCN:
        PARAMS = {  'input_vocab_size': len(INPUT_TOK.VOCAB), 
                    'embedding_size': EMBEDDING_SIZE,
                    'output_vocab_size': len(OUTPUT_TOK.VOCAB), 
                    'hidden_units': INPUT_TOK.SEQ_LENGTH,
                    'emphasize_eeg': EMPHASIZE_EEG,
                    'feedback': FEEDBACK,
                    'seq_length': INPUT_TOK.SEQ_LENGTH
                }
    elif MODEL == TransformerModel:
        PARAMS = {  'input_vocab_size': len(INPUT_TOK.VOCAB),
                    'output_vocab_size': len(OUTPUT_TOK.VOCAB),
                    'd_model': EMBEDDING_SIZE,
                    'nhead': 8,
                    'num_encoder_layers': 6,
                    'num_decoder_layers': 6,
                    'dim_feedforward': 4 * EMBEDDING_SIZE,
                    'max_seq_length': INPUT_TOK.SEQ_LENGTH
                }
    elif MODEL == MusicTransformer:
        PARAMS = {  'in_vocab_size': len(INPUT_TOK.VOCAB),
                    'out_vocab_size': len(OUTPUT_TOK.VOCAB),
                    'embedding_dim': EMBEDDING_SIZE,
                    'nhead': 4,
                    'num_layers': 3,
                    'dim_feedforward': 4 * EMBEDDING_SIZE,
                    'seq_length': INPUT_TOK.SEQ_LENGTH
                }
    else:
        raise Exception('Model not found')

    model = MODEL(**PARAMS).to(device)
    return model

def tokenize_midi_files():

    input_filenames = sorted(glob.glob(os.path.join(DATASET_PATH, 'rhythm/*.mid')))
    output_filenames = sorted(glob.glob(os.path.join(DATASET_PATH, 'melody/*.mid')))

    if FROM_MELODY_TO_RHYTHM:
        input_filenames, output_filenames = output_filenames, input_filenames

    INPUT_TOK = PrettyMidiTokenizer(TICKS_PER_BEAT=TICKS_PER_BEAT)
    OUTPUT_TOK = PrettyMidiTokenizer(TICKS_PER_BEAT=TICKS_PER_BEAT)

    for i, (in_file, out_file) in enumerate(zip(input_filenames, output_filenames)):

        in_file_name = os.path.basename(in_file)
        out_file_name = os.path.basename(out_file)
        print(f'\n{i + 1}: {in_file_name} -> {out_file_name}')

        if USE_EEG:
            if 'RELAXED' in in_file_name:
                emotion_token = BCI_TOKENS[0]
            elif 'CONCENTRATED' in in_file_name:
                emotion_token = BCI_TOKENS[1]
            else:
                raise Exception('Emotion not found in file name. Please add the emotion to the file name.')
        else:
            emotion_token = None

        in_tokens = INPUT_TOK.midi_to_tokens(in_file, 
                                             drum = not FROM_MELODY_TO_RHYTHM, 
                                             rhythm = FROM_MELODY_TO_RHYTHM,
                                             update_vocab = not COMBINE_IN_OUT_TOKENS,
                                             convert_to_integers = not COMBINE_IN_OUT_TOKENS)
        out_tokens = OUTPUT_TOK.midi_to_tokens(out_file,
                                               drum = FROM_MELODY_TO_RHYTHM,
                                               update_vocab = not COMBINE_IN_OUT_TOKENS,
                                               convert_to_integers = not COMBINE_IN_OUT_TOKENS)

        if COMBINE_IN_OUT_TOKENS:
            in_seq, out_seq = INPUT_TOK.combine_in_out_tokens(in_tokens, out_tokens, emotion_token)
            OUTPUT_TOK.VOCAB = INPUT_TOK.VOCAB
            INPUT_TOK.sequences+=in_seq
            OUTPUT_TOK.sequences+=out_seq
        else:
            in_seq = INPUT_TOK.generate_sequences(in_tokens, emotion_token)
            out_seq = OUTPUT_TOK.generate_sequences(out_tokens)

        if len(in_seq) != len(out_seq):
            min_len = min(len(INPUT_TOK.sequences), len(OUTPUT_TOK.sequences))
            INPUT_TOK.sequences = INPUT_TOK.sequences[:min_len]
            OUTPUT_TOK.sequences = OUTPUT_TOK.sequences[:min_len]

    print(f'\nNumber of input sequences: {len(INPUT_TOK.sequences)}')
    print(f'Input sequence length: {len(INPUT_TOK.sequences[0])}')
    print(f'Input sequence example: {INPUT_TOK.sequences[0]}')
    print(f'Input vocabulars size: {len(INPUT_TOK.VOCAB)}')
    print(f'\nNumber of output sequences: {len(OUTPUT_TOK.sequences)}')
    print(f'Output sequence length: {len(OUTPUT_TOK.sequences[0])}')
    print(f'Output sequence example: {OUTPUT_TOK.sequences[0]}')
    print(f'Output vocabulars size: {len(OUTPUT_TOK.VOCAB)}')

    return INPUT_TOK, OUTPUT_TOK

def initialize_dataset(train_set, eval_set, test_set):

    # Create the dataloaders
    train_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=BATCH_SIZE)

    eval_sampler = RandomSampler(eval_set)
    eval_dataloader = DataLoader(eval_set, sampler=eval_sampler, batch_size=BATCH_SIZE)

    test_sampler = RandomSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, eval_dataloader, test_dataloader

def save_model_config(model):
    path = os.path.join(RESULTS_PATH, 'config.yaml')
    with open(path, 'w') as file:
        yaml.safe_dump(model.PARAMS, file)

def save_parameters(INPUT_TOK, OUTPUT_TOK):

    # save the vocabularies
    INPUT_TOK.VOCAB.save(os.path.join(RESULTS_PATH, 'input_vocab.txt'))
    OUTPUT_TOK.VOCAB.save(os.path.join(RESULTS_PATH, 'output_vocab.txt'))

     # save the model hyperparameters in a file txt
    with open(os.path.join(RESULTS_PATH, 'parameters.txt'), 'w') as f:

        f.write(f'DATE: {time.strftime("%Y%m%d-%H%M%S")}\n')

        f.write(f'\n-----------------DATASET------------------\n')
        f.write(f'DATASET_PATH: {DATASET_PATH}\n')
        f.write(f'TRAIN_SET_SIZE: {len(train_set)}\n')
        f.write(f'EVAL_SET_SIZE: {len(eval_set)}\n')
        f.write(f'TEST_SET_SIZE: {len(test_set)}\n')

        f.write(f'\n-----------------MODEL------------------\n')
        f.write(f'MODEL: {MODEL}\n')
        f.write(f'MODEL SIZE: {MODEL_SIZE}\n')
        f.write(f'N_TOKENS: {N_TOKENS}\n')

        f.write(f'\n----------OPTIMIZATION PARAMETERS----------\n')
        f.write(f'SEED: {SEED}\n')
        f.write(f'GRADIENT_CLIP: {GRADIENT_CLIP}\n')
        f.write(f'FROM_MELODY_TO_RHYTHM: {FROM_MELODY_TO_RHYTHM}\n')
        f.write(f'COMBINE_IN_OUT_TOKENS: {COMBINE_IN_OUT_TOKENS}\n')
        f.write(f'FEEDBACK: {FEEDBACK}\n')
        f.write(f'EMPHASIZE_EEG: {EMPHASIZE_EEG}\n')
        f.write(f'LR_SCHEDULER: {LR_SCHEDULER}\n')
        f.write(f'DATA AUGMENTATION: {DATA_AUGMENTATION}\n')
        f.write(f'EARLY STOP EPOCHS: {EARLY_STOP_EPOCHS}\n')
        f.write(f'USE_EEG: {USE_EEG}\n')
        f.write(f'ENTROPY_WEIGHT: {CROSS_ENTROPY_WEIGHT}\n')
        f.write(f'PENALTY_WEIGHT: {PENALTY_WEIGHT}\n')
        f.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
        f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
        f.write(f'EPOCHS: {EPOCHS}\n')

        f.write(f'\n------------TOKENIZATION PARAMETERS--------------\n')
        f.write(f'TICKS_PER_BEAT: {TICKS_PER_BEAT}\n')
        f.write(f'TOKENS_FREQUENCY_THRESHOLD: {TOKENS_FREQUENCY_THRESHOLD}\n')
        f.write(f'SILENCE_TOKEN_WEIGHT: {SILENCE_TOKEN_WEIGHT}\n')        

def save_results():
    # plot the losses over the epochs
    plt.plot(train_losses, label='train')
    plt.plot(eval_losses, label='eval')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, 'losses.png'))
    plt.clf()

    # plot the accuracies over the epochs
    plt.plot(train_accuracies, label='train')
    plt.plot(eval_accuracies, label='eval')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, 'accuracies.png'))
    plt.clf()

    # plot the perplexities over the epochs
    plt.plot(train_perplexities, label='train')
    plt.plot(eval_perplexities, label='eval')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, 'perplexities.png'))
    plt.clf()

    with open(os.path.join(RESULTS_PATH, 'results.txt'), 'w') as f:
        f.write(f'-------------------RESULTS----------------\n')
        f.write(f'BEST_TRAIN_LOSS: {best_train_loss}\n')
        f.write(f'BEST_EVAL_LOSS: {best_eval_loss}\n')
        f.write(f'TEST_LOSS: {test_loss}\n')
        f.write(f'TRAIN_ACCURACY: {final_train_accuracy}\n')
        f.write(f'EVAL_ACCURACY: {final_eval_accuracy}\n')
        f.write(f'TEST_ACCURACY: {test_accuracy}\n')
        f.write(f'TRAIN_PERPLEXITY: {final_train_perplexity}\n')    
        f.write(f'EVAL_PERPLEXITY: {final_eval_perplexity}\n')
        f.write(f'TEST_PERPLEXITY: {test_perplexity}\n')
        f.write(f'BEST_MODEL_EPOCH: {best_model_epoch}\n')

def epoch_step(epoch, dataloader, mode):

    if mode == 'train':
        model.train()
    else:
        model.eval() # disable dropout

    total_loss = 0
    n_correct = 0
    n_total = 0

    last_output = torch.zeros([BATCH_SIZE, OUTPUT_TOK.BAR_LENGTH*3], dtype=torch.long)
    
    # iterate over the training data
    for batch_idx, (input, target) in enumerate(dataloader):

        batch_idx += 1

        # extract the emotion and the input
        emotion, input = input[:, 0].unsqueeze(1), input[:, 1:]

        # mask some tokens in the input to make the model more robust
        n_tokens_masked = random.randint(1, INPUT_TOK.BAR_LENGTH)
        mask_indices = torch.randint(3*INPUT_TOK.BAR_LENGTH, [n_tokens_masked])
        
        if USE_EEG:   
            mask = emotion.expand(-1, OUTPUT_TOK.BAR_LENGTH)
            input[:, mask_indices] = emotion
        else:
            mask = torch.zeros([input.size(0), OUTPUT_TOK.BAR_LENGTH], dtype=torch.long)
            input[:, mask_indices] = 0

        # add mask to the input last bar
        input = torch.cat((input[:, :OUTPUT_TOK.BAR_LENGTH*3], mask), dim = 1)

        # move the input and the target to the device
        input = input.to(device)
        target = target.to(device)
        last_output = last_output.to(device)

        # reset model gradients to zero
        optimizer.zero_grad()

        # Forward pass
        if MODEL == TransformerModel or MODEL == MusicTransformer: 

            steps = 0
            loss_tmp = 0
            for i in range(0, OUTPUT_TOK.BAR_LENGTH - N_TOKENS + 1, N_TOKENS):

                use_teacher_forcing = random.random() < max(0.3, 1 - (epoch / 10)) 

                # Determine if using teacher forcing or not
                if use_teacher_forcing or i == 0:
                    # Use ground truth for next input
                    input_target = target[:, i : OUTPUT_TOK.BAR_LENGTH * 3 + i]
                else:
                    # Use model's own prediction
                    input_target = last_output

                # generate the masks for the input and the target to avoid attending to future tokens
                input_mask = generate_square_subsequent_mask(input.size(1))
                input_target_mask = generate_square_subsequent_mask(input_target.size(1))

                # forward pass
                output = model(input, input_target, input_mask, input_target_mask)

                # get the last output
                last_output = torch.argmax(output, -1)

                # get the actual target
                actual_target = target[:, i + N_TOKENS : 3 * OUTPUT_TOK.BAR_LENGTH + i + N_TOKENS] # target is shifted right by one bar

                # reshape the output and the target to calculate the loss (flatten the sequences)
                actual_target = actual_target.reshape(-1) # the size -1 is inferred from other dimensions
                output = output.reshape(-1, len(OUTPUT_TOK.VOCAB))

                # calculate the loss
                loss = criterion(output, actual_target)

                if mode == 'train':
                    # calculate the gradients
                    loss.backward()

                    # clip the gradients to avoid exploding gradients
                    if GRADIENT_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

                    # update the weights
                    optimizer.step()

                loss_tmp = loss.data.item()

                # update n_correct and n_total to calculate the accuracy
                n_correct += torch.sum(torch.argmax(output, 1) == actual_target).item()
                n_total += actual_target.size(0)

                steps += 1

            loss_tmp /= steps
            total_loss += loss_tmp
            
        else:
            # forward pass
            output = model(input)

            # reshape the output and the target to calculate the loss (flatten the sequences)
            target = target.reshape(-1) # the size -1 is inferred from other dimensions
            output = output.reshape(-1, len(OUTPUT_TOK.VOCAB))

            # calculate the loss
            loss = criterion(output, target)

            if mode == 'train':
                # calculate the gradients
                loss.backward()

                # clip the gradients to avoid exploding gradients
                if GRADIENT_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

                # update the weights
                optimizer.step()

            total_loss += loss.data.item()

            # update n_correct and n_total to calculate the accuracy
            n_correct += torch.sum(torch.argmax(output, 1) == target).item()
            n_total += target.size(0)

    # calculate the accuracy
    accuracy = 100 * n_correct / n_total
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, accuracy, perplexity


def train():

    global best_eval_loss, best_train_loss, final_train_accuracy, final_eval_accuracy, final_train_perplexity, final_eval_perplexity, best_model_epoch
    global eval_losses, train_losses, train_accuracies, eval_accuracies, train_perplexities, eval_perplexities

    MODEL_PATH = os.path.join(RESULTS_PATH, 'model_state_dict.pth')

    best_eval_loss = 1e8
    best_train_loss = 1e8
    final_train_accuracy = 0
    final_eval_accuracy = 0
    final_train_perplexity = 0
    final_eval_perplexity = 0
    best_model_epoch = 0
    eval_losses = []
    train_losses = []
    train_accuracies = []
    eval_accuracies = []
    train_perplexities = []
    eval_perplexities = []

    writer = SummaryWriter(RESULTS_PATH)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=LR_PATIENCE)

    for epoch in range(1, EPOCHS+1):

        start_time = time.time()

        train_loss, train_accuracy, train_perplexity = epoch_step(epoch, train_dataloader, 'train')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)

        eval_loss, eval_accuracy, eval_perplexity = epoch_step(epoch, eval_dataloader, 'test')
        writer.add_scalar('Loss/eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/eval', eval_accuracy, epoch)
        writer.add_scalar('Perplexity/eval', eval_perplexity, epoch)

        with open(os.path.join(RESULTS_PATH, 'test_sample.txt'), 'a') as f:

            f.write(f'EPOCH: {epoch}\n')

            n_tokens_masked = random.randint(1, INPUT_TOK.BAR_LENGTH)
            mask_indices = torch.randint(3*INPUT_TOK.BAR_LENGTH, [n_tokens_masked])

            if USE_EEG:
                emotion, input = input_sample[0], input_sample[1:] 
            else:
                input = input_sample
                emotion = 0

            mask = emotion.expand(OUTPUT_TOK.BAR_LENGTH)
            input = torch.cat((input[:OUTPUT_TOK.BAR_LENGTH*3], mask))
            input[mask_indices] = emotion

            if MODEL_NAME == 'TCN':
                output = model(input.unsqueeze(0).to(device))
                output = torch.argmax(output, -1)
                f.write(f'INPUT: {input.tolist()}\n')
                f.write(f'OUTPUT: {output.tolist()}\n')
                f.write(f'TARGET: {target_sample.tolist()}\n')
            else:
                for i in range(0, OUTPUT_TOK.BAR_LENGTH - N_TOKENS + 1, N_TOKENS):
                    input_target = target_sample[i : OUTPUT_TOK.BAR_LENGTH * 3 + i]
                    output = model(input.unsqueeze(0).to(device), input_target.unsqueeze(0).to(device))
                    output = torch.argmax(output, -1)
                    actual_target = target_sample[i + N_TOKENS : 3 * OUTPUT_TOK.BAR_LENGTH + i + N_TOKENS]
                    f.write(f'STEP: {i}\n')
                    f.write(f'INPUT: {input.tolist()}\n')
                    f.write(f'OUTPUT: {output.tolist()}\n')
                    f.write(f'TARGET: {actual_target.tolist()}\n\n')
                f.write('\n')
            f.write('\n')
                    

        # Save the model if the validation loss is the best we've seen so far.
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_eval_loss = eval_loss
            final_train_accuracy = train_accuracy
            final_eval_accuracy = eval_accuracy
            final_train_perplexity = train_perplexity
            final_eval_perplexity = eval_perplexity
            best_model_epoch = epoch

        eval_losses.append(eval_loss)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        eval_accuracies.append(eval_accuracy)
        train_perplexities.append(train_perplexity)
        eval_perplexities.append(eval_perplexity)

        # Early stopping
        if epoch > EARLY_STOP_EPOCHS:
            if min(eval_losses[-EARLY_STOP_EPOCHS:]) > best_eval_loss:
                break

        # Learning rate scheduler
        if LR_SCHEDULER:
            scheduler.step(eval_loss)

        # print the loss and the progress
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print('{} | epoch {:3d}/{:3d} | lr {:02.7f} | ms/epoch {:5.5f} | train_acc {:5.2f} | eval_acc {:5.2f} | train_loss {:5.2f} | eval_loss {:5.2f} |' \
              'train_perp {:5.2f} | eval_perp {:5.2f}' \
                .format(MODEL_NAME, epoch, EPOCHS, lr, elapsed * 1000, train_accuracy, eval_accuracy, train_loss, eval_loss, train_perplexity, eval_perplexity))   

    print('\n\n TRAINING FINISHED:\n\n\tBest Loss: {:5.2f}\tBest Model saved at epoch: {:3d} \n\n' \
            .format(best_eval_loss, best_model_epoch))

    # test the model
    global test_loss, test_accuracy, test_perplexity
    test_loss, test_accuracy, test_perplexity = epoch_step(0, test_dataloader, 'eval')
    print(f'\n\nTEST LOSS: {test_loss}')
    print(f'TEST ACCURACY: {test_accuracy}')
    print(f'TEST PERPLEXITY: {test_perplexity}\n\n')

    writer.flush()
    writer.close()

if __name__ == '__main__':

    # tokenize the midi files
    INPUT_TOK, OUTPUT_TOK = tokenize_midi_files()

    # update the sequences
    print('\nUpdating INPUT_TOK sequences and vocabulary')
    INPUT_TOK.remove_less_likely_tokens(TOKENS_FREQUENCY_THRESHOLD)
    print('\nUpdating OUTPUT_TOK sequences and vocabulary')
    OUTPUT_TOK.remove_less_likely_tokens(TOKENS_FREQUENCY_THRESHOLD)

    # create the dataset
    dataset = TensorDataset(torch.LongTensor(INPUT_TOK.sequences),
                            torch.LongTensor(OUTPUT_TOK.sequences))

    # Split the dataset into training, evaluation and test sets
    train_set, eval_set, test_set = random_split(dataset, DATASET_SPLIT)
    print(f'\nTrain set size: {len(train_set)}')
    print(f'Evaluation set size: {len(eval_set)}')
    print(f'Test set size: {len(test_set)}')

    # augment the dataset
    if DATA_AUGMENTATION:
        train_set = data_augmentation_shift(train_set, [-2, -1, 1, 2])
        print(f'Training set size after augmentation: {len(train_set)}')

    # initialize the dataloaders
    print(f'\nInitializing the dataloaders...')
    global train_dataloader, eval_dataloader, test_dataloader, input_sample, target_sample
    input_sample, target_sample = train_set[0][0], train_set[0][1]
    train_dataloader, eval_dataloader, test_dataloader = initialize_dataset(train_set, eval_set, test_set)

    # initialize the model
    print(f'\nInitializing the model...')
    model = initialize_model(INPUT_TOK, OUTPUT_TOK)
    criterion = CrossEntropyWithPenaltyLoss(weight_ce=CROSS_ENTROPY_WEIGHT, weight_penalty=PENALTY_WEIGHT)
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=LEARNING_RATE)
    global MODEL_SIZE
    MODEL_SIZE = model.size()
    print(f'Model size: {MODEL_SIZE}')   

    # save the model configuration
    save_model_config(model)
    save_parameters(INPUT_TOK, OUTPUT_TOK)

    # train the model
    time.sleep(5)
    print(f'\nTraining the model...')
    train()

    # save the results
    save_results()
    