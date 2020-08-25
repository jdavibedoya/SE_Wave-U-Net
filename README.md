# Speech Enhancement using the Wave-U-Net with Spectral Losses
Adaptation of the [Wave-U-Net](https://github.com/f90/Wave-U-Net) [[1]](#1) architecture for the speech enhancement task in terms of denoising [2], dereverberation, decoloration, and bandwidth extension. The network structure is shown below.

<img src="./Wave-U-Net - Denoising + BWE.png" width="500">

## Listening Examples

## Experiments 
The experiments contemplate variations inspired by some relevant deep learning networks for speech enhancement (Spectral Losses [3][4], Noise Input Vector, Attention Mechanism [5], SEGAN+ Variations [6]). 


## Installation
### Requirements
The project is based on Python 3.6.8 and requires [libsndfile](http://mega-nerd.com/libsndfile/) and CUDA 9 to be installed. The required Python packages can be installed using ``pip install -r requirements.txt``

### Datasets
To reproduce the experiments, it is necessary to download the [VCTK](http://datashare.is.ed.ac.uk/handle/10283/1942)[7] and [DAPS](https://archive.org/details/daps_dataset) [8] datasets .


## References 
<a name="1"></a> [1] Stoller, D., Ewert, S. & Dixon, S. Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation (2018).

[2] Macartney, C. & Weyde, T. Improved Speech Enhancement with the Wave-U-Net (2018).

[3] Ramires, A., Chandna, P., Favory, X., Gómez, E. & Serra, X. Neural Percussive Synthesis Parameterised by High-Level Timbral Features. In ICASSP (2020).

[4] Feng, B., Jin, Z., Su, J. & Finkelstein, A. Learning Bandwidth Expansion UsingPerceptually-Motivated Loss. In ICASSP (2019).

[5] Giri, R., Isik, U. & Krishnaswamy, A. Attention Wave-U-Net for Speech En-hancement. In WASPAA (2019).

[6] Pascual, S., Serrà, J. & Bonafonte, A. Time-Domain Speech EnhancementUsing Generative Adversarial Networks. Speech Communication (2019).

[7] Valentini-Botinhao, C. Noisy Speech Database for Training Speech Enhance- ment Algorithms and TTS Models (2016). University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR).

[8] Mysore, G. J. Can we Automatically Transform Speech Recorded on Common Consumer Devices in Real-World Environments into Professional Production Quality Speech? - A Dataset, Insights, and Challenges. IEEE Signal Processing Letters (2015).
