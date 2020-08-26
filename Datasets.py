import Utils
import Preprocessing

import glob
import librosa
import numpy as np
import os
import random
import tensorflow as tf
from multiprocessing import Process


def take_random_snippets(sample, keys, input_shape, num_samples):
    # Take a sample (collection of audio files) and extract snippets from it at a number of random positions
    start_pos = tf.random_uniform([num_samples], 0, maxval=sample["length"] - input_shape[0], dtype=tf.int64)
    return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)


def take_all_snippets(sample, keys, input_shape, output_shape):
    # Take a sample and extract snippets from the audio signals, using a hop size equal to the output size of the network
    start_pos = tf.range(0, sample["length"] - input_shape[0], delta=output_shape[0], dtype=tf.int64)
    num_samples = start_pos.shape[0]
    return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)


def take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples):
    # Take a sample and extract snippets from the audio signals at the given start positions with the given number of samples width
    batch = dict()
    for key in keys:
        batch[key] = tf.map_fn(lambda pos: sample[key][pos:pos + input_shape[0], :], start_pos, dtype=tf.float32)
        batch[key].set_shape([num_samples, input_shape[0], input_shape[1]])
    return tf.data.Dataset.from_tensor_slices(batch)


def _floats_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_records(sample_list, model_config, input_shape, output_shape, records_path):
    # Writes samples in the given list as TFrecords into a given path, using the current model config and in/output shapes

    # Compute padding
    if (input_shape[1] - output_shape[1]) % 2 != 0:
        print("WARNING: Required number of padding of " + str(input_shape[1] - output_shape[1]) + " is uneven!")
    pad_frames = (input_shape[1] - output_shape[1]) // 2

    # Set up writers
    num_writers = 1
    writers = [tf.python_io.TFRecordWriter(records_path + str(i) + ".tfrecords") for i in range(num_writers)]

    # Go through tracks and write them to TFRecords
    keys = [model_config["input"], model_config["target"]]
    for sample in sample_list:
        try:
            audio_tracks = dict()
            audio_lengths = []
            for key in keys:
                audio, _ = Utils.load(sample[key], sr=model_config["expected_sr"], mono=True)
                audio_lengths.append(len(audio))
                audio_tracks[key] = audio
        except Exception as e:
            print(e)
            print("ERROR occurred during loading file " + str(sample) + ". Skipping")
            continue
        
        if np.abs( np.diff(audio_lengths) ) < 10: # tolerate up to 10 samples difference
            for key in keys:
              audio_tracks[key] = audio_tracks[key][:min(audio_lengths)]
              audio_tracks[key] = audio_tracks[key][:min(audio_lengths)] 

        # Pad at beginning and end with zeros
        audio_tracks = {key : np.pad(audio_tracks[key], [(pad_frames, pad_frames), (0, 0)], mode="constant", constant_values=0.0) for key in list(audio_tracks.keys())}

        # All audio tracks must be exactly same length and channels
        length = audio_tracks[model_config["input"]].shape[0]
        channels = audio_tracks[model_config["input"]].shape[1]
        for audio in list(audio_tracks.values()):
            assert(audio.shape[0] == length)
            assert (audio.shape[1] == channels)

        # Write to TFrecords the flattened version
        feature = {key: _floats_feature(audio_tracks[key]) for key in keys}
        feature["length"] = _int64_feature(length)
        feature["channels"] = _int64_feature(channels)
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        writers[np.random.randint(0, num_writers)].write(sample.SerializeToString())

    for writer in writers:
        writer.close()


def parse_record(example_proto, keys, shape):
    # Parse record from TFRecord file

    features = {key : tf.FixedLenSequenceFeature([], allow_missing=True, dtype=tf.float32) for key in keys}
    features["length"] = tf.FixedLenFeature([], tf.int64)
    features["channels"] = tf.FixedLenFeature([], tf.int64)

    parsed_features = tf.parse_single_example(example_proto, features)

    # Reshape
    length = tf.cast(parsed_features["length"], tf.int64)
    channels = tf.constant(shape[-1], tf.int64) # tf.cast(parsed_features["channels"], tf.int64)
    sample = dict()
    for key in keys:
        sample[key] = tf.reshape(parsed_features[key], tf.stack([length, channels]))
    sample["length"] = length
    sample["channels"] = channels

    return sample


def get_dataset(model_config, input_shape, output_shape, partition):
    '''
    For a model configuration and input/output shapes of the network, get the corresponding dataset for a given partition
    :param model_config: Model config
    :param input_shape: Input shape of network
    :param output_shape: Output shape of network
    :param partition: "train", "valid", or "test" partition
    :return: Tensorflow dataset object
    '''

    # Check if pre-processed dataset is already available for this model config and partition
    main_folder = model_config["data_path"] + "/VCTK_DAPS_DBE_" + model_config["input"] + "2" + model_config["target"]
    # main_folder = "/content/sample_data/" + main_folder # Data stored in Colab
    
    # Check if pre-processed dataset is already available
    if not os.path.exists(main_folder):
        # We have to prepare the datasets
        print("Preparing dataset! This could take a while...")
        
        # Prepocessing VCTK
        VCTK_preprocessed_path =  model_config["preprocessed_data_path"] + "/VCTK_8k_DBE" 
        if not os.path.exists(VCTK_preprocessed_path):
            Preprocessing.VCTK(model_config)

        # Prepocessing DAPS
        DAPS_preprocessed_path =  model_config["preprocessed_data_path"] + "/DAPS_8k_DBE" 
        if not os.path.exists(DAPS_preprocessed_path):
            Preprocessing.DAPS(model_config)
        
        # Dataset dict
        VCTK_train, VCTK_test = get_VCTK(VCTK_preprocessed_path, model_config["input"], model_config["target"])
        DAPS_train, DAPS_test = get_DAPS(DAPS_preprocessed_path, model_config["input"], model_config["target"])
        
        VCTK_random_idx_valid = np.random.choice(len(VCTK_train), size=100, replace=False)
        DAPS_random_idx_valid = np.random.choice(len(DAPS_train), size=100, replace=False)
        VCTK_valid = [VCTK_train[i] for i in VCTK_random_idx_valid]
        DAPS_valid = [DAPS_train[i] for i in DAPS_random_idx_valid]

        VCTK_train_idx = [i for i in range(len(VCTK_train)) if i not in VCTK_random_idx_valid]
        DAPS_train_idx = [i for i in range(len(DAPS_train)) if i not in DAPS_random_idx_valid]
        VCTK_train = [VCTK_train[i] for i in VCTK_train_idx]
        DAPS_train = [DAPS_train[i] for i in DAPS_train_idx]
        
        dataset = dict()
        dataset["train"] = VCTK_train + DAPS_train
        dataset["valid"] = VCTK_valid + DAPS_valid
        dataset["test"] = VCTK_test + DAPS_test
        random.shuffle(dataset["train"])

        # Convert audio files into TFRecords now
        # The dataset structure is a dictionary with "train", "valid", "test" keys, whose entries are lists, where each element represents a track.
        # Each track is represented as a dictionary containing elements "input" and "target".
        num_cores = 8

        for curr_partition in ["train", "valid", "test"]:
            print("Writing " + curr_partition + " partition...")

            # Shuffle sample order
            sample_list = dataset[curr_partition]
            random.shuffle(sample_list)

            # Create folder
            partition_folder = os.path.join(main_folder, curr_partition)
            os.makedirs(partition_folder)

            part_entries = int(np.ceil(float(len(sample_list) / float(num_cores))))
            processes = list()
            for core in range(num_cores):
                train_filename = os.path.join(partition_folder, str(core) + "_")  # address to save the TFRecords file
                sample_list_subset = sample_list[core * part_entries:min((core + 1) * part_entries, len(sample_list))]
                proc = Process(target=write_records,
                               args=(sample_list_subset, model_config, input_shape, output_shape, train_filename))
                proc.start()
                processes.append(proc)
            for p in processes:
                p.join()

    print("Dataset ready!")
    # Finally, load TFRecords dataset based on the desired partition
    dataset_folder = os.path.join(main_folder, partition)
    records_files = glob.glob(os.path.join(dataset_folder, "*.tfrecords"))
    random.shuffle(records_files)
    dataset = tf.data.TFRecordDataset(records_files)
    dataset = dataset.map(lambda x : parse_record(x, [model_config["input"], model_config["target"]], input_shape[1:]), num_parallel_calls=model_config["num_workers"])
    dataset = dataset.prefetch(10)

    # Take random samples from each track
    if partition == "train":
        dataset = dataset.flat_map(lambda x : take_random_snippets(x, [model_config["input"], model_config["target"]], input_shape[1:], model_config["num_snippets_per_track"]))
    else:
        dataset = dataset.flat_map(lambda x : take_all_snippets(x, [model_config["input"], model_config["target"]], input_shape[1:], output_shape[1:]))
    dataset = dataset.prefetch(100)

    # Cut source outputs to centre part
    dataset = dataset.map(lambda x : Utils.crop_sample(x, (input_shape[1] - output_shape[1])//2, model_config["input"])).prefetch(100)

    if partition == "train": # Repeat endlessly and shuffle when training
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=model_config["cache_size"])

    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(model_config["batch_size"]))
    dataset = dataset.prefetch(1)

    return dataset


def get_VCTK(dataset_path, input, target):
    dirs_train = set()
    dirs_test = set()
    file_names_train = set()
    file_names_test = set()
    
    for root, dirs, files in os.walk(dataset_path):
        if root == dataset_path:
          continue

        if root.split("_")[-2] == "trainset":
            dirs_train.add(root)
        elif root.split("_")[-2] == "testset":
            dirs_test.add(root)
        
        for file in files:
            if file.endswith('.wav'):
              if len(file.split("_")) > 2:
                file_name = os.path.join(root,file)
                if file_name.split("/")[-2].split("_")[-2] == "trainset":
                    file_names_train.add(file)
                if file_name.split("/")[-2].split("_")[-2] == "testset":
                    file_names_test.add(file)
    
    train = []
    test = []

    for file_name_train in file_names_train:
        samples = {}
        for dir_train in dirs_train:
            if dir_train.split("/")[-1].split("_")[0] == "noisy8k":
                samples.update({input: dir_train + os.path.sep + file_name_train})
            if dir_train.split("/")[-1].split("_")[0] == "clean":
                samples.update({target: dir_train + os.path.sep + file_name_train.replace("_8k.wav", ".wav", )})
        train.append(samples)
         
    for file_name_test in file_names_test:
        samples = {}
        for dir_test in dirs_test:
            if dir_test.split("/")[-1].split("_")[0] == "noisy8k":
                samples.update({input: dir_test + os.path.sep + file_name_test})
            if dir_test.split("/")[-1].split("_")[0] == "clean":
                samples.update({target: dir_test + os.path.sep + file_name_test.replace("_8k.wav", ".wav", )})
        test.append(samples)
        
    return [train, test]


def get_DAPS(dataset_path, input, target):
  devices_and_rooms = ['ipad_bedroom1', 'ipad_confroom1', 'ipad_confroom2', 
                       'ipad_livingroom1', 'ipad_office1', 'ipad_office2',
                       'ipadflat_confroom1', 'ipadflat_office1',
                       'iphone_bedroom1', 'iphone_livingroom1']
  train = []
  test = []
  
  clean_dir = dataset_path + "/" + target
  for root, dirs, files in os.walk(clean_dir):
    for file in files:
      if file.endswith('.wav'):
        target_file = clean_dir + "/" + file
        for device_and_room in devices_and_rooms:
          sample = {target: target_file}
          device_file = target_file.replace("clean", device_and_room)
          sample.update({input: device_file})
          if file[1:3] != "10":
            train.append(sample)
          else:
            test.append(sample)

  return [train, test]