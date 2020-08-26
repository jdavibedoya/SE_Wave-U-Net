import Evaluate

import os
from sacred import Experiment
from Config import config_ingredient


ex = Experiment('Wave-U-Net Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    output_path = None # Where to save results. Default: Same location as input.

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)