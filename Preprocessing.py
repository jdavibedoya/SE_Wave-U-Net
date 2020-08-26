import numpy as np
import shutil
import os
from essentia.standard import MonoLoader, MonoWriter, Resample


def VCTK(model_config):
    print("Preprocessing VCTK dataset")
    VCTK_path = model_config["raw_data_path"] + "/VCTK"
    VCTK_preprocessed_path =  model_config["preprocessed_data_path"] + "/VCTK_8k_DBE" 
    
    clean_dirs = ["/clean_trainset_wav", "/clean_testset_wav"]
    noisy_dirs = ["/noisy_trainset_wav", "/noisy_testset_wav"]

    # copy clean dirs
    for clean_dir in clean_dirs:
      shutil.copytree(VCTK_path + clean_dir, VCTK_preprocessed_path + clean_dir)

    # create dirs
    for noisy_dir in noisy_dirs:
        noisy8k_dir = VCTK_preprocessed_path + noisy_dir.replace("noisy", "noisy8k")
        os.makedirs(noisy8k_dir)

        # preprocessing
        for root, dirs, files in os.walk(VCTK_path + noisy_dir):
            for file in files:
                if file.endswith('.wav'):
                  # read audio
                  file_name = os.path.join(root, file)
                  noisy = MonoLoader(filename = file_name, sampleRate = 44100)()
                  noisy8k = MonoLoader(filename = file_name, sampleRate = 8000)()

                  # resample audio
                  noisy8k_resampled = Resample(inputSampleRate = 8000, outputSampleRate = 44100)(noisy8k)

                  # lengths
                  len_noisy = len(noisy)
                  len_noisy8k_resampled = len(noisy8k_resampled)

                  # trimming/appending
                  len_diff = len_noisy8k_resampled - len_noisy
                  if len_diff > 0:
                      noisy8k_resampled = noisy8k_resampled[:len_noisy]
                  elif len_diff < 0:
                      noisy8k_resampled = np.pad(noisy8k_resampled , (0,abs(len_diff)), 'constant', constant_values=(0,0))

                  # write audio        
                  output_name = noisy8k_dir + "/" + file.split(".")[0] + "_8k.wav"
                  MonoWriter(filename = output_name, sampleRate = 44100)(noisy8k_resampled)


def DAPS(model_config):
    print("Preprocessing DAPS dataset")
    DAPS_path = model_config["raw_data_path"] + "/DAPS/"
    DAPS_preprocessed_path =  model_config["preprocessed_data_path"] + "/DAPS_8k_DBE/"

    # create processed dir
    if not os.path.exists(DAPS_preprocessed_path):
        os.makedirs(DAPS_preprocessed_path)

    devices_and_rooms = ['ipad_bedroom1', 'ipad_confroom1', 'ipad_confroom2', 
                        'ipad_livingroom1', 'ipad_office1', 'ipad_office2',
                        'ipadflat_confroom1', 'ipadflat_office1',
                        'iphone_bedroom1', 'iphone_livingroom1'] #'ipad_balcony1' and 'iphone_balcony1' excluded

    duration = 5
    target = "clean"

    # Train samples
    for i in np.arange(9):
        subject = str(i + 1)
        scripts = [str(i%5 + 1), str((i+1)%5 + 1)] # 2 scripts per subject
        
        for script in scripts:
            # Female
            target_file = DAPS_path + target + "/f" + subject + "_script" + script + "_" + target + ".wav"
            target_audio = MonoLoader(filename = target_file, sampleRate = 44100)()

            starts = np.arange(0,len(target_audio),44100*duration)
            for j, start in enumerate(starts):
                # output dir
                output_dir = DAPS_preprocessed_path + target
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if len(target_audio[start:]) > (44100*duration): # check if there is a complete excerpt
                    audio_to_write = target_audio[ starts[j]: starts[j+1]]
                    output_name = output_dir + "/f" + subject + "_script" + script + "_" + target + "_" + str(j) + ".wav"
                    MonoWriter(filename = output_name, sampleRate = 44100)(audio_to_write)

            for device_and_room in devices_and_rooms:
                device_path = DAPS_path + device_and_room
                device_file = device_path + "/f" + subject + "_script" + script + "_" + device_and_room + ".wav"
                device_audio = MonoLoader(filename = device_file, sampleRate = 44100)()

                starts = np.arange(0,len(device_audio),44100*duration)
                for j, start in enumerate(starts):
                    # output dir
                    output_dir = DAPS_preprocessed_path + device_and_room
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    if len(device_audio[start:]) > (44100*duration):
                        audio_to_write = device_audio[ starts[j]: starts[j+1]]

                        audio_44k_to_8k = Resample(inputSampleRate = 44100, outputSampleRate = 8000)(audio_to_write)
                        audio_8k_to_44k = Resample(inputSampleRate = 8000, outputSampleRate = 44100)(audio_44k_to_8k)

                        len_audio_44k = len(audio_to_write)
                        len_audio_44k_resampled = len(audio_8k_to_44k)

                        # trimming/appending
                        len_diff = len_audio_44k_resampled - len_audio_44k
                        if len_diff > 0:
                            audio_8k_to_44k = audio_8k_to_44k[:len_audio_44k]
                        elif len_diff < 0:
                            audio_8k_to_44k = np.pad(audio_8k_to_44k, (0,abs(len_diff)), 'constant', constant_values=(0,0))

                        output_name = output_dir + "/f" + subject + "_script" + script + "_" + device_and_room + "_" + str(j) + ".wav"
                        MonoWriter(filename = output_name, sampleRate = 44100)(audio_8k_to_44k)

            # Male
            target_file = DAPS_path + target + "/m" + subject + "_script" + script + "_" + target + ".wav"
            target_audio = MonoLoader(filename = target_file, sampleRate = 44100)()

            starts = np.arange(0,len(target_audio),44100*duration)
            for j, start in enumerate(starts):
                # output dir
                output_dir = DAPS_preprocessed_path + target
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if len(target_audio[start:]) > (44100*duration): # check if there is a complete excerpt
                    audio_to_write = target_audio[ starts[j]: starts[j+1]]
                    output_name = output_dir + "/m" + subject + "_script" + script + "_" + target + "_" + str(j) + ".wav"
                    MonoWriter(filename = output_name, sampleRate = 44100)(audio_to_write)

            for device_and_room in devices_and_rooms:
                device_path = DAPS_path + device_and_room
                device_file = device_path + "/m" + subject + "_script" + script + "_" + device_and_room + ".wav"
                device_audio = MonoLoader(filename = device_file, sampleRate = 44100)()

                starts = np.arange(0,len(device_audio),44100*duration)
                for j, start in enumerate(starts):
                    # output dir
                    output_dir = DAPS_preprocessed_path + device_and_room
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    if len(device_audio[start:]) > (44100*duration):
                        audio_to_write = device_audio[ starts[j]: starts[j+1]]

                        audio_44k_to_8k = Resample(inputSampleRate = 44100, outputSampleRate = 8000)(audio_to_write)
                        audio_8k_to_44k = Resample(inputSampleRate = 8000, outputSampleRate = 44100)(audio_44k_to_8k)

                        len_audio_44k = len(audio_to_write)
                        len_audio_44k_resampled = len(audio_8k_to_44k)

                        # trimming/appending
                        len_diff = len_audio_44k_resampled - len_audio_44k
                        if len_diff > 0:
                            audio_8k_to_44k = audio_8k_to_44k[:len_audio_44k]
                        elif len_diff < 0:
                            audio_8k_to_44k = np.pad(audio_8k_to_44k, (0,abs(len_diff)), 'constant', constant_values=(0,0))

                        output_name = output_dir + "/m" + subject + "_script" + script + "_" + device_and_room + "_" + str(j) + ".wav"
                        MonoWriter(filename = output_name, sampleRate = 44100)(audio_8k_to_44k)

    # Test samples
    # Female
    target_file = DAPS_path + target + "/f10_script5_" + target + ".wav"
    target_audio = MonoLoader(filename = target_file, sampleRate = 44100)()

    output_name = DAPS_preprocessed_path + target + "/f10_script5_" + target + ".wav"
    MonoWriter(filename = output_name, sampleRate = 44100)(target_audio)

    for device_and_room in devices_and_rooms:
        device_path = DAPS_path + device_and_room
        device_file = device_path + "/f10_script5_" + device_and_room + ".wav"
        device_audio = MonoLoader(filename = device_file, sampleRate = 8000)()
        audio_8k_to_44k = Resample(inputSampleRate = 8000, outputSampleRate = 44100)(device_audio)

        len_audio_44k = len(target_audio)
        len_audio_44k_resampled = len(audio_8k_to_44k)

        # trimming/appending
        len_diff = len_audio_44k_resampled - len_audio_44k
        if len_diff > 0:
            audio_8k_to_44k = audio_8k_to_44k[:len_audio_44k]
        elif len_diff < 0:
            audio_8k_to_44k = np.pad(audio_8k_to_44k, (0,abs(len_diff)), 'constant', constant_values=(0,0))

        output_name = DAPS_preprocessed_path + device_and_room + "/f10_script5_" + device_and_room + ".wav"
        MonoWriter(filename = output_name, sampleRate = 44100)(audio_8k_to_44k)

    # Male
    target_file = DAPS_path + target + "/m10_script5_" + target + ".wav"
    target_audio = MonoLoader(filename = target_file, sampleRate = 44100)()

    output_name = DAPS_preprocessed_path + target + "/m10_script5_" + target + ".wav"
    MonoWriter(filename = output_name, sampleRate = 44100)(target_audio)

    for device_and_room in devices_and_rooms:
        device_path = DAPS_path + device_and_room
        device_file = device_path + "/m10_script5_" + device_and_room + ".wav"
        device_audio = MonoLoader(filename = device_file, sampleRate = 8000)()
        audio_8k_to_44k = Resample(inputSampleRate = 8000, outputSampleRate = 44100)(device_audio)

        len_audio_44k = len(target_audio)
        len_audio_44k_resampled = len(audio_8k_to_44k)

        # trimming/appending
        len_diff = len_audio_44k_resampled - len_audio_44k
        if len_diff > 0:
            audio_8k_to_44k = audio_8k_to_44k[:len_audio_44k]
        elif len_diff < 0:
            audio_8k_to_44k = np.pad(audio_8k_to_44k, (0,abs(len_diff)), 'constant', constant_values=(0,0))

        output_name = DAPS_preprocessed_path + device_and_room + "/m10_script5_" + device_and_room + ".wav"
        MonoWriter(filename = output_name, sampleRate = 44100)(audio_8k_to_44k)