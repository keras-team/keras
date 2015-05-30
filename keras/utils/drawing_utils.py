import matplotlib.pyplot as plt

class EpochDrawer(object):
    '''
    takes the history of keras.models.Sequential().fit() and
    plots training and validation loss over the epochs
    '''
    def __init__(self, history, save_filename = None):

        self.x = history['epoch']
        self.legend = ['loss']

        plt.plot(self.x, history['loss'], marker='.')

        if 'val_loss' in history:
            self.legend.append('val loss')
            plt.plot(self.x, history['val_loss'], marker='.')

        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.xticks(history['epoch'], history['epoch'])
        plt.legend(self.legend, loc = 'upper right')

        if save_filename is not None:
            plt.savefig(save_filename)

        plt.show()

