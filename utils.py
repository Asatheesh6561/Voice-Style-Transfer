import copy
import tensorflow as tf
import numpy as np
from scipy import signal
from librosa.filters import mel
from scipy.signal import get_window
from pymcd.mcd import Calculate_MCD

def butter_highpass(cutoff, fs, order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
  return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result) 
def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0

def quantize_f0_tf(x, num_bins=256):
    # x is logf0
    x = tf.reshape(tf.cast(x, tf.float32), -1)
    uv = (x <= 0)
    x = tf.where(uv, tf.zeros_like(x), x)
    x = tf.clip_by_value(x, 0, 1)
    x = tf.round(x * (num_bins - 1))
    x = x + 1
    x = tf.where(uv, tf.zeros_like(x), x)
    enc = tf.one_hot(tf.cast(x, tf.int32), num_bins+1)
    return enc, tf.cast(x, tf.int64)

def get_mask_from_lengths(lengths, max_len):
  ids = tf.range(0, max_len, dtype=tf.int64)
  mask = tf.greater_equal(ids, lengths[:, tf.newaxis])
  return mask

def pad_seq_to_2(x, len_out=128):
    sequence_length = x.shape[1]
    len_pad = len_out - sequence_length
    assert len_pad >= 0
    return tf.pad(x, [[0,0], [0,len_pad], [0,0]], 'CONSTANT'), len_pad

def get_average_mcd(first_original, second_original, first_transfer, second_transfer):
  mcd_metric = Calculate_MCD(MCD_mode="dtw")
  first_mcd = mcd_toolbox.calculate_mcd(first_original, first_transfer)
  second_mcd = mcd_toolbox.calculate_mcd(second_original, second_transfer)
  return (first_mcd + second_mcd) / 2
