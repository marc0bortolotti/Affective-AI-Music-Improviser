import glob
import numpy as np
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
from tokenization import PrettyMidiTokenizer, BCI_TOKENS, SILENCE_TOKEN, Dictionary
from architectures.transformer import TransformerModel
from architectures.tcn import TCN   
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from data_augmentation import data_augmentation_shift
from losses import CrossEntropyWithPenaltyLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\n', device)

EPOCHS = 1000 
LEARNING_RATE = 0.002 # 0.002
BATCH_SIZE = 64 # 64

ARCHITECTURES = {'transformer': TransformerModel, 'tcn' : TCN}
MODEL = ARCHITECTURES['tcn']

USE_EEG = True # use the EEG data to condition the model
FEEDBACK = True # use the feedback mechanism in the model
EMPHASIZE_EEG = False # emphasize the EEG data in the model (increase weights)
DATA_AUGMENTATION = True # augment the dataset by shifting the sequences
LR_SCHEDULER = True # use a learning rate scheduler to reduce the learning rate when the loss plateaus

TICKS_PER_BEAT = 4 
EMBEDDING_SIZE = 512 
TOKENS_FREQUENCY_THRESHOLD = None # remove tokens that appear less than # times in the dataset
SILENCE_TOKEN_WEIGHT = 0.01 # weight of the silence token in the loss function
CROSS_ENTROPY_WEIGHT = 1.0  # weight of the cross entropy loss in the total loss
PENALTY_WEIGHT = 3.0 # weight of the penalty term in the total loss (number of predictions equal to class SILENCE)

GRADIENT_CLIP = 0.35 # clip the gradients to avoid exploding gradients
DATASET_SPLIT = [0.9, 0.07, 0.03] # split the dataset into training, evaluation and test sets
EARLY_STOP_EPOCHS = 15  # stop the training if the loss does not improve for # epochs
LR_PATIENCE = 10   # reduce the learning rate if the loss does not improve for # epochs

DIRECTORY_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(DIRECTORY_PATH, f'runs/model_{time.strftime("%Y%m%d-%H%M%S")}')
DATASET_PATH = os.path.join(DIRECTORY_PATH, 'dataset')

# create a unique results path
idx = 1
while os.path.exists(RESULTS_PATH):
    RESULTS_PATH += f'_{idx}'
    idx += 1
os.makedirs(RESULTS_PATH)

def tokenize_midi_files():

    input_filenames = sorted(glob.glob(os.path.join(DATASET_PATH, 'rythms/*.mid')))
    output_filenames = sorted(glob.glob(os.path.join(DATASET_PATH, 'melodies/*.mid')))

    INPUT_TOK = PrettyMidiTokenizer()
    OUTPUT_TOK = PrettyMidiTokenizer()

    INPUT_TOK.set_ticks_per_beat(TICKS_PER_BEAT)
    OUTPUT_TOK.set_ticks_per_beat(TICKS_PER_BEAT)

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

        _ = INPUT_TOK.midi_to_tokens(in_file, update_sequences= True, update_vocab=True, emotion_token = emotion_token, instrument='drum')
        _ = OUTPUT_TOK.midi_to_tokens(out_file, update_sequences= True, update_vocab=True)

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

def initialize_dataset(train_set, eval_set, test_set):

    # Create the dataloaders
    train_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=BATCH_SIZE)

    eval_sampler = RandomSampler(eval_set)
    eval_dataloader = DataLoader(eval_set, sampler=eval_sampler, batch_size=BATCH_SIZE)

    test_sampler = RandomSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, eval_dataloader, test_dataloader

def initialize_model(INPUT_TOK, OUTPUT_TOK):

    if MODEL == TCN:
        PARAMS = {  'input_size': len(INPUT_TOK.VOCAB), 
                    'embedding_size': EMBEDDING_SIZE,
                    'output_size': len(OUTPUT_TOK.VOCAB), 
                    'hidden_units': INPUT_TOK.SEQ_LENGTH,
                    'emphasize_eeg': EMPHASIZE_EEG,
                    'feedback': FEEDBACK,
                }
    elif MODEL == TransformerModel:
        PARAMS = {  'input_size': len(INPUT_TOK.VOCAB),
                    'output_size': len(OUTPUT_TOK.VOCAB),
                    'embed_size': EMBEDDING_SIZE,
                    'nhead': 8,
                    'num_encoder_layers': 6,
                    'num_decoder_layers': 6,
                    'dim_feedforward': 4 * EMBEDDING_SIZE,
                    'seq_max_length': INPUT_TOK.SEQ_LENGTH
                }
    else:
        raise Exception('Model not found')

    model = MODEL(PARAMS).to(device)
    return model

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

        f.write(f'DATE: {time.strftime("%Y%m%d-%H%M%S")}\n\n')

        f.write(f'-----------------DATASET------------------\n')
        f.write(f'DATASET_PATH: {DATASET_PATH}\n')
        f.write(f'TRAIN_SET_SIZE: {len(train_set)}\n')
        f.write(f'EVAL_SET_SIZE: {len(eval_set)}\n')
        f.write(f'TEST_SET_SIZE: {len(test_set)}\n\n')

        f.write(f'-----------------MODEL------------------\n')
        f.write(f'MODEL SIZE: {MODEL_SIZE}\n')

        f.write(f'----------OPTIMIZATION PARAMETERS----------\n')
        f.write(f'GRADIENT_CLIP: {GRADIENT_CLIP}\n')
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
        f.write(f'EPOCHS: {EPOCHS}\n\n')

        f.write(f'------------TOKENIZATION PARAMETERS--------------\n')
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
        
def epoch_step(dataloader, mode):

    prev_output = torch.zeros([BATCH_SIZE, INPUT_TOK.SEQ_LENGTH], dtype=torch.long, device=device)

    if mode == 'train':
        model.train()
    else:
        model.eval() # disable dropout

    total_loss = 0
    n_correct = 0
    n_total = 0
    BAR_LENGTH = INPUT_TOK.BAR_LENGTH
    
    # iterate over the training data
    for batch_idx, (data, targets) in enumerate(dataloader):

        batch_idx += 1

        # mask the last bar of the input data
        batch_size = data.size(0)
        data_masked = torch.cat((data[:, :BAR_LENGTH*3], torch.zeros([batch_size, BAR_LENGTH], dtype=torch.long, device = device)), dim = 1)

        if FEEDBACK:
            input = torch.cat((data_masked, prev_output[:batch_size, :]), dim = 1)
        else:
            input = data_masked

        # reset model gradients to zero
        optimizer.zero_grad()

        tgt_input = targets[:, :-1]
        tgt_output = targets[:, 1:]

        # Forward pass
        output = model(input, tgt_input)

        # Reshape output and target to fit into the loss function
        output = output.reshape(-1, OUTPUT_VOCAB_SIZE)
        tgt_output = tgt_output.reshape(-1)

        # make the prediction
        output = model(input)[:, :INPUT_TOK.SEQ_LENGTH]

        
        prev_output = torch.argmax(output, 2)# batch, seq_len (hidden units), vocab_size

        # flatten the output sequence
        # NB: the size -1 is inferred from other dimensions
        # NB: contiguous() is used to make sure the tensor is stored in a contiguous chunk of memory, necessary for view() to work

        final_target = targets.contiguous().view(-1)
        final_output = output.contiguous().view(-1, len(OUTPUT_TOK.VOCAB))

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

        # update n_correct and n_total to calculate the accuracy
        n_correct += torch.sum(torch.argmax(final_output, 1) == final_target).item()
        n_total += final_target.size(0)

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
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=LR_PATIENCE)

    for epoch in range(1, EPOCHS+1):

        start_time = time.time()

        train_loss, train_accuracy, train_perplexity = epoch_step(train_dataloader, 'train')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)

        eval_loss, eval_accuracy, eval_perplexity = epoch_step(eval_dataloader, 'eval')
        writer.add_scalar('Loss/eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/eval', eval_accuracy, epoch)
        writer.add_scalar('Perplexity/eval', eval_perplexity, epoch)

        # Save the model if the validation loss is the best we've seen so far.
        if eval_loss < best_eval_loss:
            # torch.save(model.state_dict(), MODEL_PATH)
            best_eval_loss = eval_loss

        if train_loss < best_train_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_train_loss = train_loss
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
            if min(train_losses[-EARLY_STOP_EPOCHS:]) > best_train_loss:
                break

        # Learning rate scheduler
        if LR_SCHEDULER:
            scheduler.step(train_loss)

        # print the loss and the progress
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print('| epoch {:3d}/{:3d} | lr {:02.7f} | ms/epoch {:5.5f} | train_acc {:5.2f} | eval_acc {:5.2f} | train_loss {:5.2f} | eval_loss {:5.2f} |' \
              'train_perp {:5.2f} | eval_perp {:5.2f}' \
                .format(epoch, EPOCHS, lr, elapsed * 1000, train_accuracy, eval_accuracy, train_loss, eval_loss, train_perplexity, eval_perplexity))   


    print('\n\n TRAINING FINISHED:\n\n\tBest Loss: {:5.2f}\tBest Model saved at epoch: {:3d} \n\n' \
            .format(best_eval_loss, best_model_epoch))


    # test the model
    global test_loss, test_accuracy, test_perplexity
    test_loss, test_accuracy, test_perplexity = epoch_step(test_dataloader, 'eval')
    print(f'\n\nTEST LOSS: {test_loss}')
    print(f'TEST ACCURACY: {test_accuracy}')
    print(f'TEST PERPLEXITY: {test_perplexity}\n\n')

    writer.flush()
    writer.close()

if __name__ == '__main__':

    # tokenize the midi files
    INPUT_TOK, OUTPUT_TOK = tokenize_midi_files()

    # update the sequences
    INPUT_TOK.update_sequences(TOKENS_FREQUENCY_THRESHOLD)
    OUTPUT_TOK.update_sequences(TOKENS_FREQUENCY_THRESHOLD)

    # create the dataset
    dataset = TensorDataset(torch.LongTensor(INPUT_TOK.sequences).to(device),
                            torch.LongTensor(OUTPUT_TOK.sequences).to(device))

    # Split the dataset into training, evaluation and test sets
    train_set, eval_set, test_set = random_split(dataset, DATASET_SPLIT)
    print(f'Train set size: {len(train_set)}')
    print(f'Evaluation set size: {len(eval_set)}')
    print(f'Test set size: {len(test_set)}')

    # augment the dataset
    if DATA_AUGMENTATION:
        train_set = data_augmentation_shift(train_set, [-2, -1, 1, 2])
        print(f'Training set size after augmentation: {len(train_set)}')

    # initialize the dataloaders
    print(f'Initializing the dataloaders...')
    global train_dataloader, eval_dataloader, test_dataloader
    train_dataloader, eval_dataloader, test_dataloader = initialize_dataset(train_set, eval_set, test_set)
    
    # initialize the model
    print(f'Initializing the model...')
    model = initialize_model()
    criterion = CrossEntropyWithPenaltyLoss(weight_ce=CROSS_ENTROPY_WEIGHT, weight_penalty=PENALTY_WEIGHT)
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=LEARNING_RATE)
    global MODEL_SIZE
    MODEL_SIZE = model.size
    print(f'Model size: {MODEL_SIZE}')   

    # save the model configuration
    save_model_config(model)
    save_parameters(INPUT_TOK, OUTPUT_TOK)

    # train the model
    time.sleep(5)
    print(f'Training the model...')
    train()

    # save the results
    save_results()
    