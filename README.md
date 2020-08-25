# Speech Enhancement using the Wave-U-Net with Spectral Losses
Adaptation of the [Wave-U-Net](https://github.com/f90/Wave-U-Net) [[1]](#1) architecture for the speech enhancement task in terms of denoising [[2]](#2), dereverberation, decoloration, and bandwidth extension. The network structure is shown below.

<img src="./Wave-U-Net - Denoising + BWE.png" width="500">

## Listening Examples

## Experiments 
The experiments contemplate variations inspired by some relevant deep learning networks for speech enhancement (Spectral Losses [[3]](#3)[[4]](#4), Noise Input Vector, Attention Mechanism [[5]](#5), SEGAN+ Variations [[6]](#6)). 
| ID | Variant             | Loss            | Command                                                        |
|----|---------------------|-----------------|----------------------------------------------------------------|
| 00 | Baseline Wave-U-Net | L1 + STFT + Mel | ``python Training.py with cfg.L1``                             |
| 01 | Attention Mechanism>| L1 + STFT + Mel | ``python Training.py with cfg.AttentionMechanism_L1_STFT_Mel`` |
| 02 | ^                  .| L1 + STFT + Mel | ``python Training.py with cfg.AttentionMechanism_L1_STFT_Mel`` |
| 03 | Attention Mechanism | L1 + STFT + Mel | ``python Training.py with cfg.AttentionMechanism_L1_STFT_Mel`` |
| 04 | Attention Mechanism | L1 + STFT + Mel | ``python Training.py with cfg.AttentionMechanism_L1_STFT_Mel`` |
| 05 | Attention Mechanism | L1 + STFT + Mel | ``python Training.py with cfg.AttentionMechanism_L1_STFT_Mel`` |
| 06 | Attention Mechanism | L1 + STFT + Mel | ``python Training.py with cfg.AttentionMechanism_L1_STFT_Mel`` |
| 07 | Attention Mechanism | L1 + STFT + Mel | ``python Training.py with cfg.AttentionMechanism_L1_STFT_Mel`` |

|a  |b  |
|---|---|
|>|1|
|2||


## Installation
### Requirements
The project is based on Python 3.6.8 and requires [libsndfile](http://mega-nerd.com/libsndfile/) and CUDA 9 to be installed. The required Python packages can be installed using ``pip install -r requirements.txt``

### Datasets
To reproduce the experiments, it is necessary to download the [VCTK](http://datashare.is.ed.ac.uk/handle/10283/1942) [[7]](#7) and [DAPS](https://archive.org/details/daps_dataset) [[8]](#8) datasets .


## References 
<a name="1"></a> [1] Stoller, D., Ewert, S. & Dixon, S. Wave-U-Net: [A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185) (2018).

<a name="2"></a> [2] Macartney, C. & Weyde, T. [Improved Speech Enhancement with the Wave-U-Net](https://arxiv.org/abs/1811.11307) (2018).

<a name="3"></a> [3] Ramires, A., Chandna, P., Favory, X., Gómez, E. & Serra, X. [Neural Percussive Synthesis Parameterised by High-Level Timbral Features](https://ieeexplore.ieee.org/abstract/document/9053128). In ICASSP (2020).

<a name="4"></a> [4] Feng, B., Jin, Z., Su, J. & Finkelstein, A. [Learning Bandwidth Expansion UsingPerceptually-Motivated Loss](https://ieeexplore.ieee.org/abstract/document/8682367). In ICASSP (2019).

<a name="5"></a> [5] Giri, R., Isik, U. & Krishnaswamy, A. [Attention Wave-U-Net for Speech Enhancement](https://ieeexplore.ieee.org/abstract/document/8937186). In WASPAA (2019).

<a name="6"></a> [6] Pascual, S., Serrà, J. & Bonafonte, A. [Time-Domain Speech EnhancementUsing Generative Adversarial Networks](https://www.sciencedirect.com/science/article/abs/pii/S0167639319301359). Speech Communication (2019).

<a name="7"></a> [7] Valentini-Botinhao, C. [Noisy Speech Database for Training Speech Enhancement Algorithms and TTS Models](https://www.research.ed.ac.uk/portal/en/publications/speech-enhancement-for-a-noiserobust-texttospeech-synthesis-system-using-deep-recurrent-neural-networks(08deb6fd-79c0-490f-ae46-f37034b6bfb4).html) (2016). University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR).

<a name="8"></a> [8] Mysore, G. J. [Can we Automatically Transform Speech Recorded on Common Consumer Devices in Real-World Environments into Professional Production Quality Speech?](https://ieeexplore.ieee.org/abstract/document/6981922) - A Dataset, Insights, and Challenges. IEEE Signal Processing Letters (2015).
