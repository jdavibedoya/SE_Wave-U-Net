import librosa
import numpy as np
from pypesq import pesq as PESQ
from scipy.linalg import toeplitz


def Eval(ref, deg, sr):
    # Compute the SSNR
    ref_wav, ref_sr = librosa.load(ref, sr = sr, mono=True)
    deg_wav, deg_sr = librosa.load(deg, sr = sr, mono=True)
    
    if np.abs( len(ref_wav) -  len(deg_wav) ) < 10: # tolerate up to 10 samples difference
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = ref_wav[:min_len]
        deg_wav = deg_wav[:min_len]
    
    assert(len(ref_wav) == len(deg_wav) and ref_sr==deg_sr )
    segsnr_mean = SSNR(ref_wav, deg_wav, ref_sr)
    segSNR = np.mean(segsnr_mean)
    
    # Compute the LSD
    lsd = LSD(ref_wav, deg_wav)
        
    # Compute the PESQ
    pesq_sr = 16000
    ref_wav, ref_sr = librosa.load(ref, sr = pesq_sr, mono=True)
    deg_wav, deg_sr = librosa.load(deg, sr = pesq_sr, mono=True)

    if np.abs( len(ref_wav) -  len(deg_wav) ) < 10: # tolerate up to 10 samples difference
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = ref_wav[:min_len]
        deg_wav = deg_wav[:min_len]

    assert(len(ref_wav) == len(deg_wav) and ref_sr==deg_sr )
    pesq = PESQ(ref_wav, deg_wav, fs=pesq_sr)
    
    return pesq, lsd, segSNR, len(ref_wav)


def LSD(ref_wav, deg_wav):
	ref_S = np.log10( np.abs( librosa.core.stft(ref_wav) )**2 + 1e-10 )
	deg_S = np.log10( np.abs( librosa.core.stft(deg_wav) )**2 + 1e-10 )
	
	diff = (ref_S - deg_S)**2
	lsd = np.mean(np.sqrt(np.mean(diff, axis=0)))
	return lsd


# SSNR code taken from https://github.com/santi-pdp/segan_pytorch/blob/master/segan/utils.py   
def SSNR(ref_wav, deg_wav, srate=44100, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [P. C. Loizou, Evaluation of Objective Quality Measures for Speech Enhancement]
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    # Scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + 10e-20))

    # Global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR
    num_frames = int(clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) Get the frames for the test and ref speech
        # Apply Hanning Window
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return segmental_snr