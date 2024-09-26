import numpy as np
import torch
from torch.utils.data import TensorDataset
import re
from tokenization import SILENCE_TOKEN, BCI_TOKENS



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
    
    augmented_dataset = TensorDataset(torch.LongTensor(augmented_input_sequences), 
                                      torch.LongTensor(output_sequences))
    
    # Concatenate the original and the augmented dataset
    concatenated_dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset])

    return concatenated_dataset


def data_augmentation_transposition(dataset, transpositions, OUTPUT_TOK, probability=0.5):
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

    augmented_dataset = TensorDataset(torch.LongTensor(input_sequences), 
                                      torch.LongTensor(augmented_output_sequences))
    
    # Concatenate the original and the augmented dataset
    concatenated_dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset])

    return concatenated_dataset