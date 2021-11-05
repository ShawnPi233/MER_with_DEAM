"""
This module containg all the necessary methods for the training of CNN models to predict
values for valence and arousal.
"""
import os
import torch
import torch.nn as nn

from models import AudioNet
from data_loader import make_training_loaders
from utility_functions import *


class Trainer:
    """
    Methods for training are defined in this class.

    Attributes:
        dimension (str): specifies the type of output predicted by the model
        num_epochs (int): duration of the training process in epochs
        log_interval (int): the frequency the training progress is printed
        train_loader, test_loader: loading and batching the data in train and test sets
    """
    def __init__(self, args):

        self.dimension = args.dimension

        self._data_dir = args.data_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir
        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        self._lr = args.lr_init
        self._lr_decay = args.lr_decay
        self._weight_decay = args.weight_decay

        self.train_loader, self.test_loader = make_training_loaders(self._data_dir)

        # Get the model parameters corresponding to the dimension selected
        if self.dimension == 'valence':
            self._params_dict = args.valence_params_dict
        elif self.dimension == 'arousal':
            self._params_dict = args.arousal_params_dict
        else:
            self._params_dict = args.params_dict

        # Initialize the model
        self.model = AudioNet(self._params_dict).to(self._device)

        # Define the optimizer and the loss criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        self._criterion = nn.MSELoss()

        # Create dictionaries to save the training progress metrics in, according to the dimension selected
        if self.dimension == 'both':
            self.train_dict = {'valence_loss': [], 'arousal_loss': []}
            self.test_dict = {'valence_loss': [], 'arousal_loss': []}
        else:
            self.train_dict = {'loss': []}
            self.test_dict = {'loss': []}

    def save_model(self):
        """
        Method to save the trained model weights to a specified path.
        """
        model_path = os.path.join(self._models_dir, 'model_{:s}.pt'.format(self.dimension))
        torch.save(self.model.state_dict(), model_path)

    def update_learning_rate(self):
        """
        Method to update the learning rate, according to a decay factor.
        """
        self._lr *= self._lr_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._lr

        # Display status message
        success_message = 'Learning rate updated to {:.1e}'.format(self._lr)
        print(success_format(success_message))

    def train_1d(self):
        """
        Method to train 1D-output models. This is called when `dimension` is `valence` or `arousal`.
        """
        train_loss = []

        self.model.train()
        # Iterate over the train set
        for batch_idx, (data, annotations) in enumerate(self.train_loader):

            # Select target labels according to the `dimension`
            if self.dimension == 'valence':
                target = annotations[:, 0]
            elif self.dimension == 'arousal':
                target = annotations[:, 1]

            # Move data to device
            data = data.to(self._device)
            target = target.to(self._device)

            # Zero-out the gradients and make predictions
            self.optimizer.zero_grad()
            output = self.model(data)

            target = target.view_as(output)

            # Compute the batch loss and gradients
            batch_loss = self._criterion(output, target)
            batch_loss.backward()
            train_loss.append(batch_loss.data.cpu().numpy())

            # Update the weights
            self.optimizer.step()

        self.train_dict['loss'].append(np.array(train_loss).mean())

    def train_2d(self):
        """
        Method to train 2D-output models. This is called when `dimension` is `both`.
        """
        true_annotations = []
        pred_annotations = []

        self.model.train()
        # Iterate over the train set
        for batch_idx, (data, annotations) in enumerate(self.train_loader):

            # Move data to device
            data = data.to(self._device)
            annotations = annotations.to(self._device)

            # Zero-out the gradients and make predictions
            self.optimizer.zero_grad()
            output = self.model(data)

            # Save predicted and true annotations
            true_annotations.extend(annotations.cpu().detach().numpy())
            pred_annotations.extend(output.cpu().detach().numpy())

            # Compute the batch loss and gradients
            batch_loss = self._criterion(output, annotations)
            batch_loss.backward()

            # Update the weights
            self.optimizer.step()

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        # Extract predictions and true values for valence dimension and compute MSE
        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        # Extract predictions and true values for arousal dimension and compute MSE
        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.train_dict['valence_loss'].append(valence_mse)
        self.train_dict['arousal_loss'].append(arousal_mse)

    def validate_1d(self):
        """
        Method to validate 1D-output models. This is called when `dimension` is `valence` or `arousal`.
        """
        test_loss = []

        self.model.eval()
        # Freeze gradients
        with torch.no_grad():
            # Iterate over test set
            for batch_idx, (data, annotations) in enumerate(self.test_loader):

                # Select target labels according to the `dimension`
                if self.dimension == 'valence':
                    target = annotations[:, 0]
                elif self.dimension == 'arousal':
                    target = annotations[:, 1]

                # Move data to device
                data = data.to(self._device)
                target = target.to(self._device)

                # Make predictions
                output = self.model(data)
                target = target.view_as(output)

                # Compute batch loss
                batch_loss = self._criterion(output, target)
                test_loss.append(batch_loss.data.cpu().numpy())

        self.test_dict['loss'].append(np.array(test_loss).mean())

    def validate_2d(self):
        """
        Method to train 2D-output models. This is called when `dimension` is `both`.
        """
        true_annotations = []
        pred_annotations = []

        self.model.eval()
        # Freeze gradients
        with torch.no_grad():
            # Iterate over test set
            for batch_idx, (data, annotations) in enumerate(self.test_loader):

                # Move data to device
                data = data.to(self._device)
                annotations = annotations.to(self._device)

                # Make predictions
                output = self.model(data)

                true_annotations.extend(annotations.cpu().detach().numpy())
                pred_annotations.extend(output.cpu().detach().numpy())

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        # Extract predictions and true values for valence dimension and compute MSE
        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        # Extract predictions and true values for arousal dimension and compute MSE
        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.test_dict['valence_loss'].append(valence_mse)
        self.test_dict['arousal_loss'].append(arousal_mse)
