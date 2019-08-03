"""
From https://github.com/Bjarten/early-stopping-pytorch
"""
import numpy as np
import torch
from os import path
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, path='.', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, model, loss=None, accuracy=None):
        
        if accuracy is None:
            if loss is None:
                raise RuntimeError("No score given !")
            else:
                score = -loss
        else:
            score = accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation score increase.'''
        if self.verbose:
            print(f'Validation score increase ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        
        
        torch.save(model.state_dict(), path.join(self.path,'checkpoint.pt'))
        self.best_score = score
