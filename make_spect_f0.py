import os
import sys
import warnings
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
import tensorflow as tf
from tensorflow.python.client import device_lib
from utils import butter_highpass, speaker_normalization, pySTFT
from tqdm import tqdm
print(device_lib.list_local_devices())
np.seterr(divide='ignore', invalid='ignore')
mel_basis = mel(48000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 48000, order=5)

spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))

# Modify as needed
rootDir = 'assets/wavs'
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)
subdirList = sorted(subdirList)[75:]
for subdir in sorted(subdirList):
    print(subdir)
    
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))    
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    
    if spk2gen[subdir] == 'M':
        lo, hi = 50, 250
    elif spk2gen[subdir] == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError
    prng = RandomState(int(subdir[1:]))
    
    fileList = sorted(fileList)

    for i in tqdm(range(len(fileList))):
        x, fs = sf.read(os.path.join(dirName,subdir,fileList[i]))
        assert fs == 48000
        # x = tf.convert_to_tensor(x, dtype=tf.float32)
        # b = tf.convert_to_tensor(b, dtype=tf.float32)
        # a = tf.convert_to_tensor(a, dtype=tf.float32)
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100

        f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
        assert len(S) == len(f0_rapt)
            
        np.save(os.path.join(targetDir, subdir, fileList[i][:-4]),
                S.astype(np.float32), allow_pickle=False)    
        np.save(os.path.join(targetDir_f0, subdir, fileList[i][:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)
