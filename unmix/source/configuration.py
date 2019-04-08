#!/usr/bin/env python3
# coding: utf8

"""
Global configuration object.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import commentjson
from collections import namedtuple
import getpass
import json
import os
import shutil

from unmix.source.exceptions.configurationerror import ConfigurationError
from unmix.source.helpers import converter
from unmix.source.helpers import environmentvariables
from unmix.source.helpers import reducer


class Configuration(object):

    output_directory = ''

    @staticmethod
    def initialize(configuration_file, working_directory=None, create_output=True):
        global configuration
        if not working_directory:
            working_directory = os.getcwd()
        environmentvariables.set_environment_variables(extend=True)
        if not configuration_file:
            configuration_file = converter.env('UNMIX_CONFIGURATION_FILE')
        with open(Configuration.build_path(configuration_file), 'r') as f:
            configuration = commentjson.load(f, object_hook=lambda d: namedtuple('X', d.keys())
                                             (*map(lambda x: converter.try_eval(x), d.values())))

        Configuration.output_directory = os.path.join(working_directory, Configuration.get(
            'environment.output_path'), Configuration.get('environment.output_folder'))
        if create_output and not os.path.exists(Configuration.output_directory):
            os.makedirs(Configuration.output_directory)
        if create_output:
            Configuration.log_environment(configuration_file, working_directory)

    @staticmethod
    def log_environment(configuration_file, working_directory):
        shutil.copy(configuration_file, os.path.join(
            Configuration.output_directory, 'configuration.jsonc'))
        repo = False
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
        except:
            pass
        enviornment = {
            "time": converter.get_timestamp(),
            "data_collection": Configuration.get('collection.folder'),
            "configuration_file": configuration_file,
            "working_directory": working_directory,
            "user": getpass.getuser(),
            "repository": repo.working_dir if repo else "",
            "branch": repo.active_branch.name if repo else "",
            "commit": repo.head.object.hexsha if repo else "",
            "variables": dict(os.environ)
        }
        with open(os.path.join(Configuration.output_directory, 'enviornment.json'), 'w') as file:
            json.dump(enviornment, file, indent=4)

    @staticmethod
    def get(key='', optional=True):
        global configuration
        if not configuration:
            raise ConfigurationError()
        if key:
            try:
                value = reducer.rgetattr(configuration, key)
                if not optional and not value:
                    raise ConfigurationError(key)
                return value
            except:
                raise ConfigurationError(key)
        return key

    @staticmethod
    def get_path(key='', create=True, optional=True):
        path = Configuration.get(key, optional)
        return Configuration.build_path(path)

    @staticmethod
    def build_path(path, create=True):
        """
        Generates an absolute path if a relative is passed.
        """
        if not os.path.isabs(path):
            path = os.path.join(Configuration.output_directory, path)
        if create and not os.path.exists(path):
            os.makedirs(path)
        return path
