"""
This module contains functions of general utility used in multiple modules.
"""
import numpy as np


def get_quadrant(measurements, value_range=(0, 1)):
    """
    Function to get the quadrant localization from valence-arousal annotation.
    :param measurements: tuple/list of valence and arousal values
    :param value_range: range the values for valence and arousal are in
    :return: quadrant number (1, 2, 3 or 4)
    """
    # Find the range middle value
    val_min, val_max = value_range
    val_mid = (val_max - val_min) / 2 + val_min

    # Get quadrant number
    valence, arousal = measurements
    if valence > val_mid:
        return 1 if arousal > val_mid else 4
    else:
        return 2 if arousal >= val_mid else 3


def get_quadrants_dict(valence_dict, arousal_dict):
    """
    Function to extract quadrants information from valence and arousal data.
    :param valence_dict: dictionary with valence information
    :param arousal_dict: dictionary with arousal information
    :return: dictionary with quadrants information
    """

    # Get the true and predicted annotations from valence and arousal
    true_annotations = [measurement for measurement in zip(valence_dict['true_annotations'],
                                                           arousal_dict['true_annotations'])]
    pred_annotations = [measurement for measurement in zip(valence_dict['pred_annotations'],
                                                           arousal_dict['pred_annotations'])]

    # Get quadrants from annotations
    true_quadrant = np.array([get_quadrant(measurement) for measurement in true_annotations])
    pred_quadrant = np.array([get_quadrant(measurement) for measurement in pred_annotations])

    quadrants_dict = {'true_annotations': true_quadrant,
                      'pred_annotations': pred_quadrant}

    quadrants_names = [1, 2, 3, 4]
    for quadrant in quadrants_names:
        # Compute quadrants accuracy
        q_pred = true_quadrant[np.where(pred_quadrant == quadrant)]
        q_true = true_quadrant[np.where(true_quadrant == quadrant)]
        q_perc = np.sum(q_pred == quadrant) / len(q_true) * 100
        quadrants_dict['{:d}'.format(quadrant)] = q_perc

    return quadrants_dict


def scale_measurement(measurement, current_range, new_range=(0, 1)):
    """
    Function to scale measurement to a new range.
    :param measurement: measurement to be rescaled
    :param current_range: the range the measurement is in
    :param new_range: the new value range the measurement will be scaled to
    :return: rescaled measurement
    """
    old_min, old_max = current_range
    new_min, new_max = new_range
    measurement_scaled = (measurement - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return measurement_scaled


def print_epoch(epoch, train_dict, validation_dict, dimension):
    """
    Function to format the display of epoch log.
    :param epoch: the epoch to print information about
    :param train_dict: dictionary with information about training process
    :param validation_dict: dictionary with information about validation process
    :param dimension: the dimension the predictions are made for
    """

    # Set the print messages according to the `dimension`
    if dimension == 'both':
        print_keys = {'valence_loss': 'Valence',
                      'arousal_loss': 'Arousal'}
    else:
        print_keys = {'loss': dimension.capitalize()}

    print('\nEPOCH {:d}'.format(epoch))
    print('-' * 35)

    # Print train information
    print('   Train loss')
    for key, message in print_keys.items():
        print('  {:>12s} : {:.3f}'.format(message, train_dict[key][-1]))

    # Print validation information
    if validation_dict is not None:
        print('\n   Validation loss')
        for key, message in print_keys.items():
            print('  {:>12s} : {:.3f}'.format(message, validation_dict[key][-1]))


def print_test_results(valence_dict, arousal_dict, quadrants_dict):
    """
    Function to display testing information.
    :param valence_dict: dictionary with information about valence dimension
    :param arousal_dict: dictionary with information about arousal dimension
    :param quadrants_dict: dictionary with information about quadrants
    """

    # Print metrics for valence dimension
    print('VALENCE')
    print('   MAE : {:.4f}'.format(valence_dict['mae']))
    print('   MSE : {:.4f}'.format(valence_dict['mse']))
    print()
    # Print metrics for arousal dimension
    print('AROUSAL')
    print('   MAE : {:.4f}'.format(arousal_dict['mae']))
    print('   MSE : {:.4f}'.format(arousal_dict['mse']))
    print()
    # Print metrics for quadrants
    print('Accuracy')
    for quadrant in range(1, 5):
        print('   QUADRANT {:d} : {:.2f}%%'.format(quadrant, quadrants_dict[quadrant]))


def print_params(args_dict):
    """
    Function to format the display of command line arguments dictionary.
    :param args_dict: dictionary of command line arguments
    """
    for key, val in args_dict.items():
        print(f'{key} : {val}')


def fail_format(fail_message):
    """
    Function to format the display of failed operation information.
    :param fail_message: information message to print
    """
    fail_flag = '===FAILED==='

    return "\n{:s}\n   {:s}\n{:s}\n".format(fail_flag, fail_message, fail_flag)


def success_format(success_message):
    """
    Function to format the display of successful operation information.
    :param success_message: information message to print
    """
    success_flag = '===SUCCEEDED==='

    return "\n{:s}\n   {:s}\n{:s}\n".format(success_flag, success_message, success_flag)

