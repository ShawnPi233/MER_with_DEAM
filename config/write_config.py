"""
Writes arguments in JSON dictionary.
"""
import json

args_dict = {'data_dir': {'type': 'str',
                          'default': 'Data',
                          'help': 'path to data directory'
                          },
             'deam_dir': {'type': 'str',
                          'default': 'Data/DEAM_dataset',
                          'help': 'path to DEAM raw data directory'
                          },
             'font_dir': {'type': 'str',
                          'default': 'Font',
                          'help': 'path to directory where the font used in plotting is stored'
                          },
             'models_dir': {'type': 'str',
                            'default': 'Models',
                            'help': 'path to directory where trained models are written'
                            },
             'plots_dir': {'type': 'str',
                           'default': 'Plots',
                           'help': 'path to directory where figures are written'
                           },
             'audio_extension': {'type': 'str',
                                 'default': 'mp3',
                                 'help': 'extension of audio excerpts fom DEAM dataset'
                                 },
             'sample_rate': {'type': 'int',
                             'default': 44100,
                             'help': 'sample rate for loading and preprocessing audio data'
                             },
             'device': {'type': 'str',
                        'default': 'cuda',
                        'help': 'use CUDA if available'
                        },
             'mode': {'type': 'str',
                      'default': 'train',
                      'help': 'train | test | preprocess - training / testing mode or to preprocess data'
                      },
             'dimension': {'type': 'str',
                           'default': 'both',
                           'help': 'both | valence | arousal - train a model to predict values for both '
                                   'valence & arousal or separate models to predict in each dimension'
                           },
             'params_dict': {'type': 'json.loads',
                             'default': '{"in_ch": 20, '
                                        '"num_filters1": 32, '
                                        '"num_filters2": 64, '
                                        '"num_hidden": 128, '
                                        '"out_size": 2}',
                             'help': 'layers parameters for the 2D-output model'
                             },
             'valence_params_dict': {'type': 'json.loads',
                                     'default': '{"in_ch": 20, '
                                                '"num_filters1": 32, '
                                                '"num_filters2": 64, '
                                                '"num_hidden": 64, '
                                                '"out_size": 1}',
                                     'help': 'layers parameters for the valence-output model'
                                     },
             'arousal_params_dict': {'type': 'json.loads',
                                     'default': '{"in_ch": 20, '
                                                '"num_filters1": 32, '
                                                '"num_filters2": 32, '
                                                '"num_hidden": 64, '
                                                '"out_size": 1}',
                                     'help': 'layers parameters for the arousal-output model'
                                     },
             'lr_init': {'type': 'float',
                         'default': 1e-3
                         },
             'lr_decay': {'type': 'float',
                          'default': 1e-1
                          },
             'decay_interval': {'type': 'int',
                                'default': 1000
                                },
             'weight_decay': {'type': 'float',
                              'default': 1e-2
                              },
             'num_epochs': {'type': 'int',
                            'default': 2000
                            },
             'log_interval': {'type': 'int',
                              'default': 1
                              }
             }

with open('config_file.json', 'w') as outfile:
    json.dump(args_dict, outfile)
