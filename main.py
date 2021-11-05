import os

from config.config import ArgsFromJSON

from data_preprocessing import DataPreprocessor
from visualize import Visualizer
from train import Trainer
from test import Tester
from utility_functions import *


def preprocess_data(args):
    """
    Function to process the data and create data sets.
    :param args: command line arguments
    """

    # Create directories for font and plots if they do not exist
    if not os.path.exists(args.font_dir):
        os.mkdir(args.font_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    # Create objects for preprocessing and visualization
    data_preprocessor = DataPreprocessor(args)
    visualizer = Visualizer(args.font_dir, args.plots_dir)

    # Get information from DEAM files
    data_preprocessor.get_data_info()
    # Get waveforms from audio
    data_preprocessor.get_waveforms()
    # Augment dataset
    data_preprocessor.augment_quadrants()
    # Create sets for train and test
    data_preprocessor.make_train_test_sets()

    # Visualize data distribution in quadrants
    visualizer.visualize_data_distribution(data_preprocessor.annotations, data_preprocessor.quadrants)
    # Visualize data distribution on each valence and arousal dimensions
    visualizer.visualize_dimensions_distribution(data_preprocessor.annotations)


def run_train(args):
    """
    Function to train a model.
    :param args: command line arguments
    """
    # Create directories for font, models and plots if they do not exist
    if not os.path.exists(args.font_dir):
        os.mkdir(args.font_dir)
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    # Create objects for training and visualization
    visualizer = Visualizer(args.font_dir, args.plots_dir)
    trainer = Trainer(args)

    for epoch in range(trainer.num_epochs):

        # Train and validate a model to predict values for both valence and arousal
        if trainer.dimension == 'both':
            trainer.train_2d()
            trainer.validate_2d()

        # Train and validate a model to predict values for valence or arousal, according to `dimension`
        else:
            trainer.train_1d()
            trainer.validate_1d()

        # Display epoch every `log_interval`
        if (epoch + 1) % trainer.log_interval == 0 or (epoch + 1) == trainer.num_epochs:
            print_epoch(epoch + 1, trainer.train_dict, trainer.test_dict, trainer.dimension)

        # Update the learing rate every `decay_interval`
        if (epoch + 1) % args.decay_interval == 0:
            trainer.update_learning_rate()

    # Visualize train and validation losses
    visualizer.plot_losses(trainer.train_dict, trainer.test_dict, trainer.dimension)
    # Save the trained model
    trainer.save_model()


def run_test(args):
    """
    Function for testing a model.
    :param args: command line arguments
    """

    # Create directories for plots if it doesn't exist
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    # Create objects for testing and visualization
    visualizer = Visualizer(args.font_dir, args.plots_dir)
    tester = Tester(args)

    # Test a model that predicts values for both valence and arousal
    if tester.dimension == 'both':
        tester.load_model_2d()
        tester.test_2d()

    # Test models that separately predict values for valence and arousal, respectively
    else:
        tester.load_model_1d()
        tester.test_1d()

    if tester.dimension == 'both':
        title = '2D Model'
    else:
        title = '1D Models'

    # Visualize quadrant predictions
    visualizer.plot_quadrant_predictions(tester.valence_dict, tester.arousal_dict, tester.quadrants_dict, title)
    # Visualize valence predictions
    visualizer.plot_valence_predictions(tester.valence_dict, title)
    # Visualize arousal predictions
    visualizer.plot_arousal_predictions(tester.arousal_dict, title)


if __name__ == '__main__':

    # Read args from JSON dictionary
    args_from_json = ArgsFromJSON('config/config_file.json')
    args_from_json.get_args_from_dict()
    args = args_from_json.parser.parse_args()

    print('\n\n')
    print_params(vars(args))
    print('\n\n')

    if args.mode == 'preprocess':
        preprocess_data(args)

    elif args.mode == 'train':
        run_train(args)

    elif args.mode == 'test':
        run_test(args)
