"""
This module containg all the necessary methods for testing the models trained to predict values
for valence and arousal.
"""
import os
import torch

from models import AudioNet
from data_loader import make_testing_loader
from utility_functions import *


class Tester:
    """
    Methods for testing are defined in this class.

    Attributes:
        dimension (str): specifies the type of output predicted by the model
        test_loader: loading and batching the data in test set
        model: AudioNet model with parameters according to `dimension`
        valence_dict, arousal_dict, quadrants_dict: dictionaries containing data needed for computing
        performance metrics and visualization
    """
    def __init__(self, args):

        self.dimension = args.dimension

        self._data_dir = args.data_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir
        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.test_loader = make_testing_loader(self._data_dir)

        if self.dimension == 'both':
            self._params_dict = args.params_dict
            self.model = AudioNet(self._params_dict).to(self._device)
        else:
            self._valence_params_dict = args.valence_params_dict
            self._arousal_params_dict = args.arousal_params_dict

            self.valence_model = AudioNet(self._valence_params_dict).to(self._device)
            self.arousal_model = AudioNet(self._arousal_params_dict).to(self._device)

        self.valence_dict = dict()
        self.arousal_dict = dict()
        self.quadrants_dict = dict()

    def load_model_1d(self):
        """
        Method to load the pretrained models to predict separately values for valence and arousal dimension,
        respectively.
        """
        valence_path = os.path.join(self._models_dir, 'model_valence.pt')
        self.valence_model.load_state_dict(torch.load(valence_path))

        arousal_path = os.path.join(self._models_dir, 'model_arousal.pt')
        self.arousal_model.load_state_dict(torch.load(arousal_path))

    def load_model_2d(self):
        """
        Method to load the pretrained model to predict values for both valence and arousal dimensions.
        """
        model_path = os.path.join(self._models_dir, 'model_{:s}.pt'.format(self.dimension))
        self.model.load_state_dict(torch.load(model_path))

    def test_1d(self):
        """
        Method to test 1D-output models. This is called when `dimension` is `valence` or `arousal`.
        """
        true_valence = []
        pred_valence = []
        true_arousal = []
        pred_arousal = []

        self.valence_model.eval()
        self.arousal_model.eval()
        # Freeze gradients
        with torch.no_grad():
            for batch_idx, (data, annotations) in enumerate(self.test_loader):

                # Create individual target labels for each dimension
                valence_target = annotations[:, 0]
                arousal_target = annotations[:, 1]

                # Move data to device
                data = data.to(self._device)
                valence_target = valence_target.to(self._device)
                arousal_target = arousal_target.to(self._device)

                # Make predictions for valence
                valence_output = self.valence_model(data)
                valence_target = valence_target.view_as(valence_output)

                # Make predictions for arousal
                arousal_output = self.arousal_model(data)
                arousal_target = arousal_target.view_as(arousal_output)

                true_valence.extend(valence_target.cpu().detach().squeeze().numpy())
                pred_valence.extend(valence_output.cpu().detach().squeeze().numpy())

                true_arousal.extend(arousal_target.cpu().detach().squeeze().numpy())
                pred_arousal.extend(arousal_output.cpu().detach().squeeze().numpy())

        true_valence, pred_valence = np.array(true_valence), np.array(pred_valence)

        # Compute valence MAE and MSE
        valence_mae = np.mean(np.abs(true_valence - pred_valence))
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        # Compute arousal MAE and MSE
        true_arousal, pred_arousal = np.array(true_arousal), np.array(pred_arousal)
        arousal_mae = np.mean(np.abs(true_arousal - pred_arousal))
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.valence_dict['true_annotations'] = true_valence
        self.valence_dict['pred_annotations'] = pred_valence
        self.valence_dict['mae'] = valence_mae
        self.valence_dict['mse'] = valence_mse

        self.arousal_dict['true_annotations'] = true_arousal
        self.arousal_dict['pred_annotations'] = pred_arousal
        self.arousal_dict['mae'] = arousal_mae
        self.arousal_dict['mse'] = arousal_mse

        # Extract information about quadrants from valence & arousal annotations
        self.quadrants_dict = get_quadrants_dict(self.valence_dict, self.arousal_dict)

    def test_2d(self):
        """
        Method to test 2D-output models. This is called when `dimension` is `both`.
        """

        true_annotations = []
        pred_annotations = []

        self.model.eval()
        # Freeze gradients
        with torch.no_grad():
            for batch_idx, (data, annotations) in enumerate(self.test_loader):

                # Move data to device
                data = data.to(self._device)
                annotations = annotations.to(self._device)

                # Make predictions for valence and arousal
                output = self.model(data)

                true_annotations.extend(annotations.cpu().detach().numpy())
                pred_annotations.extend(output.cpu().detach().numpy())

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        # Compute valence MAE and MSE
        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mae = np.mean(np.abs(true_valence - pred_valence))
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        # Extract predictions and true values for valence and arousal dimensions and compute MAE and MSE
        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mae = np.mean(np.abs(true_arousal - pred_arousal))
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.valence_dict['true_annotations'] = true_valence
        self.valence_dict['pred_annotations'] = pred_valence
        self.valence_dict['mae'] = valence_mae
        self.valence_dict['mse'] = valence_mse

        self.arousal_dict['true_annotations'] = true_arousal
        self.arousal_dict['pred_annotations'] = pred_arousal
        self.arousal_dict['mae'] = arousal_mae
        self.arousal_dict['mse'] = arousal_mse

        # Extract information about quadrants from valence & arousal annotations
        self.quadrants_dict = get_quadrants_dict(self.valence_dict, self.arousal_dict)
