import Datasets
import Evaluate
import Models.UnetAudioEnhancer
import Test
import Utils

import functools
import numpy as np
import os
import tensorflow as tf
from Config import config_ingredient
from sacred import Experiment
from tensorflow.contrib.signal.python.ops import window_ops


ex = Experiment('Wave-U-Net - Training', ingredients=[config_ingredient])

@ex.config
# Executed for training, sets the seed value to the Sacred config so that Sacred fixes the Python and Numpy RNG to the same state everytime.
def set_seed():
    seed = 1337

@config_ingredient.capture
def train(model_config, experiment_id, load_model=None):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    enhancer_class = Models.UnetAudioEnhancer.UnetAudioEnhancer(model_config)

    enhancer_input_shape, enhancer_output_shape = enhancer_class.get_padding(np.array(disc_input_shape))
    enhancer_func = enhancer_class.get_output

    # Placeholders and input normalisation
    dataset = Datasets.get_dataset(model_config, enhancer_input_shape, enhancer_output_shape, partition="train")
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    print("Training...")

    # BUILD MODEL
    # Enhancer
    enhancer_output = enhancer_func(batch[model_config["input"]], True, reuse=False)

    # Supervised objective
    enhancer_loss = 0
    target = batch[model_config["target"]]
    if model_config["raw_audio_loss"]:
        enhancer_loss += tf.reduce_mean(tf.abs(target - enhancer_output)) # L1

    if model_config["stft_loss"]:
      stft_loss = Utils.stft_loss(enhancer_output, target, sample_rate=model_config["expected_sr"], frame_length=model_config["frame_length_stft"])
      enhancer_loss += model_config["stft_loss"] * tf.reduce_mean(stft_loss)

    if model_config["perceptual_loss"]:
      perceptual_loss = Utils.perceptual_loss(enhancer_output, target, sample_rate=model_config["expected_sr"], frame_length=model_config["frame_length_perceptual"])
      enhancer_loss += model_config["perceptual_loss"] * tf.reduce_mean(perceptual_loss)
    
    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)

    # Set up optimizers
    enhancer_vars = Utils.getTrainableVariables("enhancer")
    print("Enhancer_Vars: " + str(Utils.getNumParams(enhancer_vars)))
    print("Num of variables: " + str(len(tf.global_variables())))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("enhancer_solver"):
            enhancer_solver = tf.train.AdamOptimizer(learning_rate=model_config["init_sup_enhancer_lr"]).minimize(enhancer_loss, var_list=enhancer_vars)

    # SUMMARIES
    tf.summary.scalar("enhancer_loss", enhancer_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')

    # Start session and queue input threads
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep + str(experiment_id),graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        print("Num of variables: " + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file: ' + load_model)

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    # Start training loop
    _global_step = sess.run(global_step)
    _init_step = _global_step
    for _ in range(model_config["epoch_it"]):
        # TRAIN ENHANCER
        _, _sup_summaries = sess.run([enhancer_solver, sup_summaries])
        writer.add_summary(_sup_summaries, global_step=_global_step)

        # Increment step counter, check if maximum iterations per epoch is achieved and stop in that case
        _global_step = sess.run(increment_global_step)

    # Epoch finished - Save model
    print("Finished epoch!")
    save_path = saver.save(sess, model_config["model_base_dir"] + os.path.sep + str(experiment_id) + os.path.sep + str(experiment_id), global_step=int(_global_step))

    # Close session, clear computational graph
    writer.flush()
    writer.close()
    sess.close()
    tf.reset_default_graph()

    return save_path


@config_ingredient.capture
def optimise(model_config, experiment_id):
    epoch = 0
    best_loss = 100000
    model_path = None
    best_model_path = None
    for i in range(2):
        worse_epochs = 0
        if i==1:
            worse_epochs = 0
            print("Finished first round of training, now entering fine-tuning stage")
            model_config["batch_size"] *= 2
            model_config["init_sup_enhancer_lr"] = 1e-5
        while worse_epochs < model_config["worse_epochs"]: # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            model_path = train(load_model=model_path) 
            curr_loss = Test.test(model_config, model_folder=str(experiment_id), partition="valid", load_model=model_path)
            epoch += 1
            if curr_loss < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + best_model_path)
    test_loss = Test.test(model_config, model_folder=str(experiment_id), partition="test", load_model=best_model_path)
    return best_model_path, test_loss


@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    experiment_id = cfg["experiment_id"]
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
           os.makedirs(dir)

    # Optimize in a supervised fashion until validation loss worsens
    sup_model_path, sup_loss = optimise()
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
    
    # Evaluate trained model
    Evaluate.produce_estimates(model_config, "VCTK", str(experiment_id), sup_model_path) # VCTK
    Evaluate.produce_estimates(model_config, "DAPS", str(experiment_id), sup_model_path) # DAPS
    