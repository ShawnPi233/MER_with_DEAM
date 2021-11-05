"""
This module contains methods for converting a JSON dictionary to command line arguments.
"""
import argparse
import json


class ArgsFromJSON:
    def __init__(self, config_file_path):
        """
        :param config_file_path: path to JSON dictionary
        """

        # Load the dictionary
        with open(config_file_path) as infile:
            self.config_dict = json.load(infile)

        self.types_dict = {"str": str,
                           "int": int,
                           "float": float,
                           "bool": bool,
                           "json.loads": json.loads}
        self.parser = argparse.ArgumentParser()

    def get_args_from_dict(self):

        for name, val in self.config_dict.items():
            arg_name = '--{:s}'.format(name)
            if "type" in val.keys() and "default" in val.keys() and "help" in val.keys():
                arg_type = self.types_dict[val["type"]]
                arg_default = arg_type(val["default"]) if val["type"] != "json.loads" else '{:s}'.format(val["default"])
                arg_help = val["help"]

                self.parser.add_argument(arg_name, type=arg_type, default=arg_default, help=arg_help)

            elif "type" in val.keys() and "default" in val.keys():
                arg_type = self.types_dict[val["type"]]
                arg_default = arg_type(val["default"]) if val["type"] != "json.loads" else '{:s}'.format(val["default"])

                self.parser.add_argument(arg_name, type=arg_type, default=arg_default)

            elif "type" in val.keys():
                arg_type = self.types_dict[val["type"]]

                self.parser.add_argument(arg_name, type=arg_type)

            elif "default" in val.keys():
                arg_default = arg_type(val["default"]) if val["type"] != "json.loads" else '{:s}'.format(val["default"])

                self.parser.add_argument(arg_name, default=arg_default)
