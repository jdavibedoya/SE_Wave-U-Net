import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    # Base configuration
    model_config = {"raw_data_path" : "data/audio/raw",
                    "preprocessed_data_path" : "data/audio/preprocessed",
                    "input" : "device8k",
                    "target" : "clean",
                    "data_path" : "data", # Set this to where the preprocessed dataset should be saved
                    "estimates_path" : "estimates", # Set this path to where you want estimates produced by the trained model to be saved
                    "model_base_dir" : "checkpoints", # Base folder for model checkpoints
                    "log_dir" : "logs", # Base folder for logs files
                    "batch_size" : 16, # Batch size
                    "init_sup_enhancer_lr" : 1e-4, # Supervised enhancer learning rate
                    "epoch_it" : 2000, # Number of supervised enhancer steps per epoch
                    'cache_size': 10800, # Number of audio snippets buffered in the random shuffle queue. Larger is better, since workers put multiple examples of one track into this queue. The number of different tracks that is sampled from with each batch equals cache_size / num_snippets_per_track. Set as high as your RAM allows.
                    'num_workers' : 4, # Number of processes used for each TF map operation used when loading the dataset
                    "num_snippets_per_track" : 5, # Number of snippets that should be extracted from each track at a time after loading it. Higher values make data loading faster, but can reduce the batches track diversity
                    'num_layers' : 12, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of convolution in the output layer
                    'num_initial_filters' : 24, # Number of filters for convolution in first layer of network
                    "num_frames": 16384, # DESIRED number of time frames in the output waveform per samples
                    'expected_sr': 44100,  # Resample all audio input to this sampling rate
                    'num_channels': 1, # Mono audio input
                    'output_activation' : 'tanh', # Activation function for output layer. "tanh" or "linear". Linear output involves clipping to [-1,1] at test time, and might be more stable than tanh
                    'padding': "valid", # Convolution is only performed on the available input, thus the output is smaller than the input
                    'upsampling' : 'learned', # Type of technique used for upsampling the feature maps in a U-Net architecture, either 'linear' interpolation or 'learned' filling in of extra samples
                    'worse_epochs' : 20, # Patience for early stoppping on validation set
                    'raw_audio_loss' : True, # L1
                    "stft_loss" : 0,
                    "frame_length_stft" : 4096,
                    "perceptual_loss" : 0,
                    "frame_length_perceptual" : 4096,
                    "noise_input_vector" : False,
                    "attention_mechanism" : False,
                    'scalar_skip_connection' : False,
                    "skipping_post_activation" : True,
                    "z_latent" : False,
                    }


@config_ingredient.named_config
def L1():
    print("Baseline")
    experiment_id = 300

@config_ingredient.named_config
def L1_STFT():
    print("L1 + STFT")
    experiment_id = 301
    model_config = {
        'stft_loss' : 1,
    }

@config_ingredient.named_config
def L1_Mel():
    print("L1 + Mel")
    experiment_id = 302
    model_config = {
        'perceptual_loss' : 1,
    }

@config_ingredient.named_config
def STFT():
    print("STFT")
    experiment_id = 303
    model_config = {
        'raw_audio_loss' : False,
        'stft_loss' : 1,
    }

@config_ingredient.named_config
def Mel():
    print("Mel")
    experiment_id = 304
    model_config = {
        'raw_audio_loss' : False,
        'perceptual_loss' : 1,
    }

@config_ingredient.named_config
def STFT_Mel():
    print("STFT + Mel")
    experiment_id = 305
    model_config = {
        'raw_audio_loss' : False,
        'stft_loss' : 1,
        'perceptual_loss' : 1,
    }

@config_ingredient.named_config
def L1_STFT_Mel():
    print("L1 + STFT + Mel")
    experiment_id = 306
    model_config = {
        'stft_loss' : 1,
        'perceptual_loss' : 1,
    }

@config_ingredient.named_config
def NoiseInputVector_L1():
    print("Noise Input Vector - L1")
    experiment_id = 307
    model_config = {
        "noise_input_vector" : True,
    }

@config_ingredient.named_config
def NoiseInputVector_L1_STFT():
    print("Noise Input Vector - L1 + STFT")
    experiment_id = 308
    model_config = {
        'stft_loss' : 1,
        "noise_input_vector" : True,
    }

@config_ingredient.named_config
def NoiseInputVector_L1_Mel():
    print("Noise Input Vector - L1 + Mel")
    experiment_id = 309
    model_config = {
        'perceptual_loss' : 1,
        "noise_input_vector" : True,
    }

@config_ingredient.named_config
def NoiseInputVector_STFT():
    print("Noise Input Vector - STFT")
    experiment_id = 310
    model_config = {
        'raw_audio_loss' : False,
        'stft_loss' : 1,
        "noise_input_vector" : True,
    }

@config_ingredient.named_config
def NoiseInputVector_Mel():
    print("Noise Input Vector - Mel")
    experiment_id = 311
    model_config = {
        'raw_audio_loss' : False,
        'perceptual_loss' : 1,
        "noise_input_vector" : True,
    }

@config_ingredient.named_config
def NoiseInputVector_STFT_Mel():
    print("Noise Input Vector - STFT + Mel")
    experiment_id = 312
    model_config = {
        'raw_audio_loss' : False,
        'stft_loss' : 1,
        'perceptual_loss' : 1,
        "noise_input_vector" : True,
    }

@config_ingredient.named_config
def L1_NoiseInputVector():
    print("Noise Input Vector - L1 + STFT + Mel")
    experiment_id = 313
    model_config = {
        'stft_loss' : 1,
        'perceptual_loss' : 1,
        "noise_input_vector" : True,
    }

@config_ingredient.named_config
def AttentionMechanism_L1():
    print("Attention Mechanism - L1")
    experiment_id = 314
    model_config = {
        "attention_mechanism" : True,
    }

@config_ingredient.named_config
def AttentionMechanism_L1_STFT_Mel():
    print("Attention Mechanism - L1 + STFT + Mel")
    experiment_id = 315
    model_config = {
        'stft_loss' : 1,
        'perceptual_loss' : 1,
        "attention_mechanism" : True,
    }

@config_ingredient.named_config
def SEGANVariations_L1():
    print("SEGAN+ Variations - L1")
    experiment_id = 316
    model_config = {
        'scalar_skip_connection' : True,
        "skipping_post_activation" : False,
        "z_latent" : True,
    }

@config_ingredient.named_config
def SEGANVariations_L1_STFT_Mel():
    print("SEGAN+ Variations - L1 + STFT + Mel")
    experiment_id = 317
    model_config = {
        'stft_loss' : 1,
        'perceptual_loss' : 1,
        'scalar_skip_connection' : True,
        "skipping_post_activation" : False,
        "z_latent" : True,
    }