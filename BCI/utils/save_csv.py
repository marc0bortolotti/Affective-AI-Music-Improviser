import numpy as np

def save_csv(path, data):
    """
    Save data to a csv file
    :param path: path to the file
    :param data: data to be saved in Mne RawArray format
    :return: None
    """
    
    arr = data.get_data()

    # rescale 
    arr *= 1e6

    # save csv
    np.savetxt(path, arr.T, delimiter=',')

