# SSTF
A Tensorflow implementation of the Voice Style Transfer paper: Unsupervised Speech Decomposition Via Triple Information Bottleneck.

## Dependencies
 - Python 3.6
 - Numpy
 - Scipy
 - Tensorflow >= 1.4.1
 - librosa
 - pysptk
 - soundfile
 - matplotlib
 - wavenet_vocoder: ```pip install wavenet_vocoder == 0.1.1```

## Training
Download [training data](https://datashare.ed.ac.uk/handle/10283/2651) from the CSTR VCTK corpus to ```assets```.
1. Extract spectrogram and f0: ```python make_spect_f0.py```
2. Generate training metadata: ```python make_metadata.py```
3. Run training scripts: ```python main.py```

