import numpy as np
import torch
from torch.utils.data import TensorDataset


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
