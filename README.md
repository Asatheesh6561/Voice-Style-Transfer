# SSTF: Speech Split TensorFlow
SSTF (Speech Split TensorFlow) is a voice style transfer project built using TensorFlow. It enables the transformation of speech signals from one voice style to another while maintaining the content. This is done by learning a mapping between different speaker styles and applying it to transfer voice characteristics such as pitch, timbre, and speech rhythm.


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

