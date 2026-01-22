"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Parameters.
"""

import os
import yaml


class Settings:
    def __init__(self, settings_yaml):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- habituation ---
            habituation = settings['habituation']
            self.decay_rate = habituation['decay_rate']
            self.decay_rate_si = habituation['decay_rate_si']
            self.top_neurons = habituation['top_neurons']
            self.top_neurons_si = habituation['top_neurons_si']

            # --- si ---
            si = settings['si']
            self.strength_regulator = si['strength_regulator']

            # --- dataset ---
            dataset = settings['dataset']
            self.dataset_name = dataset['name']
            assert self.dataset_name in ["NCALTECH12", "NCALTECH256", "NCALTECH101", "NMNIST","eventSym"]

            # --- setup ---
            setup = settings['setup']
            self.scenario = setup['scenario']
            self.seed = setup['seed']
            self.number_seeds = setup['number_seeds']
            self.tasks = setup['tasks']
            self.iterations = setup['iterations']
            self.batch_size = setup['batch_size']
            self.gating_proportion_decoder = setup['gating_proportion_decoder']
            self.z_dimension = setup['z_dimension']
            self.batch_normalization = setup['batch_normalization']
            self.learning_rate = setup['learning_rate']



