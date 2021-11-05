"""
This module contains all the necessary methods for data pre-processing.
"""
import os
import csv
import random

import librosa
from sklearn.utils import shuffle

from utility_functions import *


def get_audio_mfccs(wave, sample_rate):
    """
    Function to crop an audio waveform and extract MFCCs features.
    For more information please refer to https://gloria-m.github.io/unimodal.html#s0

    :param wave: waveform of an audio
    :param sample_rate: the rate the audio was sampled at
    :return: MFCCs features of size 20x1200
    """

    # The initial duration of the waveform is 45sec.
    # The features are extracted from an excerpt of 36sec duration.
    full_length = 45 * sample_rate
    crop_length = 36 * sample_rate

    # The windows length is 30ms
    sr_ms = sample_rate / 1000
    win_length = int(30 * sr_ms)

    diff_length = full_length - crop_length

    # Select a random point in the wave duration to represent the cropped sample start time
    # Crop 36sec starting from the selected point
    crop_start = np.random.randint(diff_length, size=1)[0]
    crop_end = crop_start + crop_length
    sample = wave[crop_start:crop_end]

    # Extract MFCCs features from non-overlapping windows of 30ms length
    sample_mfcc = librosa.feature.mfcc(sample, sr=sample_rate, n_mfcc=20,
                                       n_fft=win_length, hop_length=win_length)

    return sample_mfcc


class DataPreprocessor:
    """
    Methods for dataset preprocessing are defined in this class.
    """
    def __init__(self, args):

        self._data_dir = args.data_dir
        self._deam_dir = args.deam_dir
        self._audio_dir = os.path.join(self._deam_dir, 'Audio')
        self._annotations_path = os.path.join(self._deam_dir, 'static_annotations.csv')

        self._waves_dir = os.path.join(self._deam_dir, 'Waveforms')
        if not os.path.exists(self._waves_dir):
            os.mkdir(self._waves_dir)

        self._audio_extension = args.audio_extension
        self._sample_rate = args.sample_rate

        self.audio_names = []
        self.annotations = []
        self.quadrants = []

        self.train_audio_names = []
        self.train_annotations = []
        self.test_audio_names = []
        self.test_annotations = []

        self.train_mfccs = []
        self.test_mfccs = []

    def get_data_info(self):
        """
        Method to extract information from the annotations file provided in the DEAM dataset.

        Create lists of corresponding audio names, rescaled valence - arousal annotations and quadrants
        """

        # The annotations provided are in range 1..9
        initial_range = (1, 9)

        with open(self._annotations_path, newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)

            # File structure: song_id, valence_mean, valence_std, arousal_mean, arousal_std
            for line in reader:
                self.audio_names.append(line[0])

                # Scale down the annotations to range 0..1
                initial_valence, initial_arousal = float(line[1]), float(line[3])
                scaled_valence = scale_measurement(initial_valence, initial_range)
                scaled_arousal = scale_measurement(initial_arousal, initial_range)
                measurements = [scaled_valence, scaled_arousal]
                self.annotations.append(measurements)

                # Get the quadrant corresponding to the valence-arousal annotations
                self.quadrants.append(get_quadrant(measurements))

        self.audio_names = np.array(self.audio_names)
        self.annotations = np.array(self.annotations)
        self.quadrants = np.array(self.quadrants)

        # Display dataset information
        print('\nDEAM Dataset : {:d} samples'.format(len(self.audio_names)))
        for quadrant in range(4):
            quadrant_count = np.sum(self.quadrants == quadrant + 1)
            print('  Quadrant {:d} : {:d} samples'.format(quadrant + 1, quadrant_count))

    def get_waveforms(self):
        """
        Method to get and save waveforms from audio resampled at 44,100Hz/sec and extended or shortened
        to 45sec duration.
        """

        sr_ms = self._sample_rate / 1000
        for idx, audio_name in enumerate(self.audio_names):

            # Load and resample the audio sample
            audio_path = os.path.join(self._audio_dir, '{:s}.{:s}'.format(audio_name, self._audio_extension))
            wave, _ = librosa.load(audio_path, self._sample_rate)

            # Get the duration in miliseconds
            duration = len(wave) / sr_ms
            # If the duration is smaller than 45000ms, extend the sample by appending the last portion of the
            # waveform to the end.
            if duration < 45000:
                diff = int((duration - 45000) * sr_ms)
                wave = np.concatenate([wave, wave[diff:]])

            # If the duration is llarger than 45000ms, keep only the first 45000ms and drop the
            # last portion of the waveform
            else:
                wave = wave[:45*self._sample_rate]

            # Save the waveform as numpy array with the audio sample name
            wave_path = os.path.join(self._waves_dir, '{:s}.npy'.format(audio_name))
            np.save(wave_path, wave)

    def augment_quadrants(self):
        """
        Method to augment the datatset according to the distribution in the four quadrants.

        Setting the `desired_size` to 500 samples, in the case of over-represented quadrants, samples are randomly
        dropped to reach the desired size, while in the case of under-represented quadrants, the samples are randomly
        duplicated to reach the desired size.

        You can read more about the augmentation method at https://gloria-m.github.io/unimodal.html#s7.
        """

        desired_size = 500
        quadrant_names = [1, 2, 3, 4]

        for q_name in quadrant_names:

            # Get information of quadrant `q_name`
            q_idxs = np.where(self.quadrants == q_name)[0]
            q_size = len(q_idxs)
            print('\nQUADRANT {:d} : {:>4d} samples'.format(q_name, q_size))

            # If the quadrant size is larger than the desired_size, select randomly the samples to be kept
            if q_size >= desired_size:
                q_augmented_idxs = q_idxs[np.array(random.sample(range(q_size), desired_size))]
                q_audio_names = self.audio_names[q_augmented_idxs]
                q_annotations = self.annotations[q_augmented_idxs]

                print('    Choosing {:>4d} samples'.format(desired_size))
                print('   Resulting {:>4d} samples'.format(len(q_audio_names)))

            # If the quadrant size is smaller than the desired_size, duplicate samples until the desired_size is reached
            else:
                augm_size = desired_size - q_size
                q_augmented_idxs = q_idxs[np.random.randint(q_size, size=augm_size)]
                q_audio_names = np.concatenate([self.audio_names[q_idxs], self.audio_names[q_augmented_idxs]])
                q_annotations = np.concatenate([self.annotations[q_idxs], self.annotations[q_augmented_idxs]])

                print('     Keeping {:>4d} samples'.format(q_size))
                print('    Choosing {:>4d} samples'.format(augm_size))
                print('   Resulting {:>4d} samples'.format(len(q_audio_names)))

            # Create the train set with the first 400 samples in each quadrant
            # Create the test set with the last 100 samples in each quadrant
            self.train_audio_names.extend(list(q_audio_names)[:400])
            self.train_annotations.extend(list(q_annotations)[:400])
            self.test_audio_names.extend(list(q_audio_names)[400:])
            self.test_annotations.extend(list(q_annotations)[400:])

        # Shuffle the data for better training
        self.train_audio_names, self.train_annotations = shuffle(self.train_audio_names, self.train_annotations)
        self.test_audio_names, self.test_annotations = shuffle(self.test_audio_names, self.test_annotations)

        # Save the augmented and shuffled audio_names and the corresponding annotations, for easy reuse
        np.save(os.path.join(self._data_dir, 'train_audio_names.npy'), self.train_audio_names)
        np.save(os.path.join(self._data_dir, 'train_annotations.npy'), self.train_annotations)
        np.save(os.path.join(self._data_dir, 'test_audio_names.npy'), self.test_audio_names)
        np.save(os.path.join(self._data_dir, 'test_annotations.npy'), self.test_annotations)

    def make_train_test_sets(self):
        """
        Method to extract MFCCs features from audio.

        Load the audio waveform stored in `.npy` format and extract MFCCs by calling the function `get_audio_mfccs()
        defined above.
        Save the extracted MFCCs features from the samples in train and test sets.
        """

        # Extract MFCCs features from train data
        for idx, audio_name in enumerate(self.train_audio_names):
            # Load the sample waveform
            wave_path = os.path.join(self._waves_dir, '{:s}.npy'.format(audio_name))
            wave = librosa.load(wave_path, self._sample_rate)

            # Extract MFCCs features
            mfcc = get_audio_mfccs(wave, self._sample_rate)
            self.train_mfccs.append(mfcc)

        # Extract MFCCs features from test data
        for idx, audio_name in enumerate(self.test_audio_names):
            # Load the sample waveform
            wave_path = os.path.join(self._waves_dir, '{:s}.npy'.format(audio_name))
            wave = librosa.load(wave_path, self._sample_rate)

            # Extract MFCCs features
            mfcc = get_audio_mfccs(wave, self._sample_rate)
            self.test_mfccs.append(mfcc)

        # Save the extracted audio features
        np.save(os.path.join(self._data_dir, 'train_mfccs.npy'), self.train_mfccs)
        np.save(os.path.join(self._data_dir, 'test_mfccs.npy'), self.test_mfccs)
