import Evaluate

import os
from sacred import Experiment
from Config import config_ingredient


ex = Experiment('Wave-U-Net Prediction', ingredients=[config_ingredient])

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_estimate(model_config, model_path, input_path, output_path)
