import Datasets
import Models.UnetAudioEnhancer
import Utils

import functools
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.signal.python.ops import window_ops


def test(model_config, partition, model_folder, load_model):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    enhancer_class = Models.UnetAudioEnhancer.UnetAudioEnhancer(model_config)

    enhancer_input_shape, enhancer_output_shape = enhancer_class.get_padding(np.array(disc_input_shape))
    enhancer_func = enhancer_class.get_output

    # Creating the batch generators
    assert ((enhancer_input_shape[1] - enhancer_output_shape[1]) % 2 == 0)
    dataset = Datasets.get_dataset(model_config, enhancer_input_shape, enhancer_output_shape, partition=partition)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    print("Testing...")

    # BUILD MODEL
    # Enhancer
    enhancer_output = enhancer_func(batch[model_config["input"]], False, reuse=False)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep +  model_folder, graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to test
    restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    print("Num of variables: " + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for testing')

    # Start training loop
    _global_step = sess.run(global_step)
    print("Starting!")

    total_loss = 0.0
    batch_num = 1

    # Supervised objective
    enhancer_loss = 0
    target = batch[model_config["target"]]
    if model_config["raw_audio_loss"]: # L1
        enhancer_loss += tf.reduce_mean(tf.abs(target - enhancer_output))

    if model_config["stft_loss"]:
      stft_loss = Utils.stft_loss(enhancer_output, target, sample_rate=model_config["expected_sr"], frame_length=model_config["frame_length_stft"])
      enhancer_loss += model_config["stft_loss"] * tf.reduce_mean(stft_loss)

    if model_config["perceptual_loss"]:
      perceptual_loss = Utils.perceptual_loss(enhancer_output, target, sample_rate=model_config["expected_sr"], frame_length=model_config["frame_length_perceptual"])
      enhancer_loss += model_config["perceptual_loss"] * tf.reduce_mean(perceptual_loss)
        
    while True:
        try:
            curr_loss = sess.run(enhancer_loss)
            total_loss = total_loss + (1.0 / float(batch_num)) * (curr_loss - total_loss)
            batch_num += 1
        except tf.errors.OutOfRangeError as e:
            break

    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=total_loss)])
    writer.add_summary(summary, global_step=_global_step)

    writer.flush()
    writer.close()

    print("Finished testing - Total loss: " + str(total_loss))

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

    return total_loss