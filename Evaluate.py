import Datasets
import Metrics
import Models.UnetAudioEnhancer
import Utils

import csv
import librosa
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from essentia.standard import MonoLoader, MonoWriter, Resample


def predict(track_audio, model_config, load_model):
    '''
    Model has to be saved beforehand into a pickle file containing model configuration dictionary and checkpoint path!
    :param track: Audio (n_samples, n_channels)
    :param model_config: Model config
    :param load_model: Model checkpoint path
    :return: Prediction
    '''

    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    enhancer_class = Models.UnetAudioEnhancer.UnetAudioEnhancer(model_config)

    enhancer_input_shape, enhancer_output_shape = enhancer_class.get_padding(np.array(disc_input_shape))
    enhancer_func = enhancer_class.get_output

    # Batch size of 1
    enhancer_input_shape[0] = 1
    enhancer_output_shape[0] = 1

    input_ph = tf.placeholder(tf.float32, enhancer_input_shape)

    print("Evaluating...")

    # BUILD MODELS
    # Enhancer
    enhancer_output = enhancer_func(input_ph, training=False, reuse=False)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Load pretrained model
    restorer = tf.train.Saver(None, write_version=tf.train.SaverDef.V2)
    print("Num of variables: " + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for track prediction')

    enhancer_pred = predict_track(model_config, sess, track_audio, enhancer_input_shape, enhancer_output_shape, enhancer_output, input_ph)

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()
    
    return enhancer_pred


def predict_track(model_config, sess, track_audio, enhancer_input_shape, enhancer_output_shape, enhancer_output, input_ph):
    '''
    Outputs estimate for a given input audio signal [n_frames, n_channels] and a given Tensorflow session and placeholders belonging to the prediction network.
    It iterates through the track, collecting segment-wise predictions to form the output.
    :param model_config: Model configuration dictionary
    :param sess: Tensorflow session used to run the network inference
    :param track_audio: [n_frames, n_channels] audio signal (numpy array). Can have different sampling rate or channels than the model supports, will be resampled correspondingly.
    :param enhancer_input_shape: Input shape of enhancer ([batch_size, num_samples, num_channels])
    :param enancer_output_shape: Output shape of enhancer ([batch_size, num_samples, num_channels])
    :param enhancer_ouput: Tensorflow tensor that represents the output of the network
    :param input_ph: Input tensor of the network
    :return: Prediction
    '''
    
    # Append zeros to track_audio if its shorter than input size of network - this will be cut off at the end again
    if track_audio.shape[0] < enhancer_input_shape[1]:
        extra_pad = enhancer_input_shape[1] - track_audio.shape[0]
        track_audio = np.pad(track_audio, [(0, extra_pad), (0,0)], mode="constant", constant_values=0.0)
    else:
        extra_pad = 0
    
    # Pre-allocate prediction (same shape as input)
    track_time_frames = track_audio.shape[0]
    prediction = np.zeros(track_audio.shape, np.float32)

    input_time_frames = enhancer_input_shape[1]
    output_time_frames = enhancer_output_shape[1]

    # Pad track_audio across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_time_frames = (input_time_frames - output_time_frames) // 2
    track_audio_padded = np.pad(track_audio, [(pad_time_frames, pad_time_frames), (0,0)], mode="constant", constant_values=0.0)

    # Iterate over track_audio magnitudes, fetch network prediction
    for pos in range(0, track_time_frames, output_time_frames):
        # If this output patch would reach over the end of the track, set it so we predict the very end of the output, then stop
        if pos + output_time_frames > track_time_frames:
            pos = track_time_frames - output_time_frames

        # Prepare track_audio excerpt by selecting time interval
        track_part = track_audio_padded[pos:pos + input_time_frames,:]
        track_part = np.expand_dims(track_part, axis=0)

        prediction_part = sess.run(enhancer_output, feed_dict={input_ph: track_part})

        # Save prediction
        # prediction_shape = [1, output_time_frames, num_chan]
        prediction[pos:pos + output_time_frames] = prediction_part[0, :, :]

    # In case we had to pad the track_audio at the end, remove those samples from the prediction now
    if extra_pad > 0:
        prediction = prediction[:-extra_pad,:]
    
    return prediction


def evaluate_dataset(model_config, dataset, experiment_id, load_model):
    '''
    For a given input file, saves the prediction made by a given model.
    :param model_config: Model configuration
    :param dataset: Dataset to evaluate
    :param experiment_id: ID of the experiment
    :param load_model: Model checkpoint path
    '''
    
    output_path = model_config["estimates_path"] + os.path.sep + experiment_id + "_" + dataset

    csv_file_name = model_config["estimates_path"] + os.path.sep + experiment_id + "_" + dataset + ".csv" 
    with open(csv_file_name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["target_file", "output_file", "pesq", "lsd", "ssnr", "audio_len"])
        
        # Get test set
        if dataset == "VCTK":
            _, test = Datasets.get_VCTK(model_config["preprocessed_data_path"] + "/VCTK_8k_DBE", model_config["input"], model_config["target"])
        elif dataset == "DAPS":
            _, test = Datasets.get_DAPS(model_config["preprocessed_data_path"] + "/DAPS_8k_DBE", model_config["input"], model_config["target"])

        for sample in test:
            file = sample[model_config["input"]].split("/")[-1]
            print("Producing estimate for file " + file)
            track_audio, sr = Utils.load(sample[model_config["input"]], sr = model_config['expected_sr'], mono=True)
            prediction_audio = predict(track_audio, sr, model_config, load_model) # Get estimate
            prediction_file_name = os.path.join(output_path, file) + "_prediction.wav"
            
            # Save estimate as audio file
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            librosa.output.write_wav(prediction_file_name, prediction_audio, sr)
            
            # Evaluate
            target_file_name = sample[model_config["target"]]
            pesq, lsd, ssnr, audio_len = Metrics.Eval(target_file_name, prediction_file_name, model_config['expected_sr'])
            csv_writer.writerow([target_file_name, prediction_file_name, pesq, lsd, ssnr, audio_len])
            print('PESQ:{:.3f} LSD:{:.3f} SSNR:{:.3f}'.format(pesq, lsd, ssnr))   

              
    results_df = pd.read_csv(csv_file_name, usecols=["pesq", "lsd", "ssnr", "audio_len"])
    pesq, lsd, ssnr = (results_df[["pesq", "lsd", "ssnr"]].multiply(results_df["audio_len"], axis=0)).sum()/sum(results_df["audio_len"])
    print('Results -> PESQ:{:.3f} LSD:{:.3f} SSNR:{:.3f}'.format(pesq, lsd, ssnr))


def produce_estimate(model_config, model_path, input_path, output_path):
    print("Producing estimate for file " + input_path)

    # Read audio
    audio = MonoLoader(filename = input_path, sampleRate = model_config['expected_sr'])()
    audio_8k = MonoLoader(filename = input_path, sampleRate = 8000)()

    # Resample audio
    audio_nb = Resample(inputSampleRate = 8000, outputSampleRate = 44100)(audio_8k)

    # Lengths
    len_audio = len(audio)
    len_audio_nb = len(audio_nb)

    # Trimming/appending
    len_diff = len_audio_nb - len_audio
    if len_diff > 0:
      audio_nb = audio_nb[:len_audio]
    elif len_diff < 0:
      audio_nb = np.pad(audio_nb, (0,abs(len_diff)), 'constant', constant_values=(0,0))

    # Prediction
    audio_nb = np.expand_dims(audio_nb, axis=0).T #(n_frames, n_channels)
    prediction_audio = predict(audio_nb, model_config, model_path) # Get estimate
    prediction_file_name = os.path.join(output_path, input_path.split("/")[-1]) + "_prediction.wav"

    # Save estimate as audio file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    librosa.output.write_wav(prediction_file_name, prediction_audio, model_config['expected_sr'])


def noisy(model_config, dataset):
    csv_file_name = model_config["estimates_path"] + os.path.sep + "noisy" + "_" + dataset +  ".csv"
    with open(csv_file_name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["target_file", "input_file", "pesq", "lsd", "ssnr", "audio_len"])
        
        # Get test set
        if dataset == "VCTK":
            _, test = Datasets.get_VCTK(model_config["preprocessed_data_path"] + "/VCTK_8k_DBE", model_config["input"], model_config["target"])
        elif dataset == "DAPS":
            _, test = Datasets.get_DAPS(model_config["preprocessed_data_path"] + "/DAPS_8k_DBE", model_config["input"], model_config["target"])

        for sample in test:
            # Evaluate
            input_file_name = sample[model_config["input"]]
            print("Test file " + input_file_name)
            target_file_name = sample[model_config["target"]]
            pesq, lsd, ssnr, audio_len = Metrics.Eval(target_file_name, input_file_name, model_config['expected_sr'])
            csv_writer.writerow([target_file_name, input_file_name, pesq, lsd, ssnr, audio_len])
            print('PESQ:{:.3f} LSD:{:.3f} SSNR:{:.3f}'.format(pesq, lsd, ssnr))
            
    results_df = pd.read_csv(csv_file_name, usecols=["ssnr", "lsd", "pesq", "audio_len"])
    pesq, lsd, ssnr = (results_df[["pesq", "lsd", "ssnr"]].multiply(results_df["audio_len"], axis=0)).sum()/sum(results_df["audio_len"])
    print('Results -> PESQ:{:.3f} LSD:{:.3f} SSNR:{:.3f}'.format(pesq, lsd, ssnr))
