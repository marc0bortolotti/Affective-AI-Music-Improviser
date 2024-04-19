import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import numpy as np


def plot_feature_vector(x, x_flat, epoch=1):
    """
    Plot the feature vector and the original signal
    :param x: numpy array of shape (n_epochs, n_channels, n_samples)
    :param x_flat: numpy array of shape (n_epochs, n_channels * n_samples)
    :param epoch: int, index of the epoch to plot
    """
    plt.plot(x[epoch, 0], label='Channel 1', color='blue')
    offset = np.arange(200, 400)
    plt.plot(offset, x[epoch, 1], label='Channel 2', color='red')
    offset = np.arange(400, 600)
    plt.plot(offset, x[epoch, 2], label='Channel 3', color='orange')
    offset = np.arange(600, 800)
    plt.plot(offset, x[epoch, 3], label='Channel 4', color='purple')
    offset = np.arange(800, 1000)
    plt.plot(offset, x[epoch, 4], label='Channel 5', color='brown')
    offset = np.arange(1000, 1200)
    plt.plot(offset, x[epoch, 5], label='Channel 6', color='pink')
    offset = np.arange(1200, 1400)
    plt.plot(offset, x[epoch, 6], label='Channel 7', color='gray')
    offset = np.arange(1400, 1600)
    plt.plot(offset, x[epoch, 7], label='Channel 8', color='black')

    plt.plot(x_flat[epoch, :], label='Feature Vector', color='green', linestyle='dotted', linewidth=2)
    plt.legend(loc='upper right')
    plt.xlim(0, 1600)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.show()


def plot_data_distribution(x_train, x_test):
    """
    Plot the distribution of the mean of the feature vector
    :param x_train: numpy array of shape (n_epochs, n_channels, n_samples)
    :param x_test: numpy array of shape (n_epochs, n_channels, n_samples)
    """
    plt.hist(np.mean(x_train, axis=0), bins=30, alpha=0.5, label='Train', color='green')
    plt.hist(np.mean(x_test, axis=0), bins=30, alpha=0.3, label='Test', color='blue')
    plt.legend(loc='upper right')
    plt.xlabel('Mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of the mean of the feature vector')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, save_data=False, save_path=None, classifier=None):
    """
    Plot the confusion matrix
    :param y_true: numpy array of shape (n_samples, )
    :param y_pred: numpy array of shape (n_samples, )
    :param classes: list of class names
    :param normalize: bool, whether to normalize the confusion matrix or not
    :param title: str, title of the plot
    :param cmap: matplotlib colormap
    """

    title= ' Confusion Matrix'
    cmap=plt.cm.Blues

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = ' - Normalized ' + title

    title = classifier + title

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
   
    if save_data:
        plt.savefig(save_path + '/' + classifier + '_confusion_matrix.png')

    # plt.show()
    plt.close()
    
  


def plot_cross_validated_confusion_matrix(X, y, clf, cv, classes=None, normalize=False, title='Cross-Validated Confusion Matrix',
                                          cmap=plt.cm.Blues, classifier=None, save_data=False, save_path=None):
    """ Plot the cross-validated confusion matrix
    :param X: numpy array of shape (n_samples, n_features)
    :param y: numpy array of shape (n_samples, )
    :param clf: classifier object
    :param cv: cross-validation method
    :param classes: list of class names
    :param normalize: bool, whether to normalize the confusion matrix or not
    :param title: str, title of the plot
    :param cmap: matplotlib colormap
    """
    # Perform cross-validated predictions
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title

    title = classifier + ' - ' + title
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path + '/' + classifier + '_cross_validated_confusion_matrix.png')
    # plt.show()


def plot_points_scatter(X, y, labels, save_data=False, save_path=None, classifier=None):

    plt.figure()
    colors = ["turquoise", "darkorange", "red", "green", "blue", "yellow", "purple", "pink", "gray"]
    target_names = labels
    lw = 2

    # extract indexes for each class in y_train
    idx_classes = []
    for i in range(len(labels)):
        idx_classes.append(np.where(y == i))

    for i, x in enumerate(idx_classes):
        X = np.squeeze(X)

        if len(X.shape) > 1 and classifier != 'SVM':
            plt.scatter(X[x, 0], X[x, 1], color=colors[i], alpha=0.8, lw=lw, label=target_names[i])
        elif len(X.shape) < 2: 
            plt.scatter(x, X[x], color=colors[i], alpha=0.8, lw=lw, label=target_names[i])

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title(classifier + ' - Scatter Plot')

    if save_data:
        plt.savefig(save_path + '/' + classifier + '_scatter_plot.png')

    # plt.show()
    plt.close()