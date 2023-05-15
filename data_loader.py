import os 
import tensorflow as tf
import pickle  
import numpy as np
import hparams
from random import randint
import time
from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager
class SpeechSplitDataset():
    """Dataset class for the Utterances Dataset"""

    def __init__(self, root_dir, feat_dir, mode):
        """Initialize and preprocess the Utterances Dataset"""
        self.root_dir = root_dir
        self.feat_dir = feat_dir
        self.mode = mode
        self.step = 20
        self.split = 0
        self.min_len_seq = hparams.min_len_seq
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad
        metaname = os.path.join(self.root_dir, "train.pkl")
        meta = pickle.load(open(metaname, "rb"))
        
        self.dataset = []
        for i in range(0, len(meta), self.step):
            submeta = meta[i:i+self.step]
            uttrs = [self.load_data(sbmt, mode) for sbmt in submeta]
            self.dataset.extend(uttrs)
        if mode == 'train':
            self.train_dataset = self.dataset
            self.num_tokens = len(self.train_dataset)
        elif mode == 'test':
            self.test_dataset = self.dataset
            self.num_tokens = len(self.test_dataset)
        else:
            raise ValueError
        self.num_samples = self.num_tokens
        self.n_repeats = hparams.samplier
        self.shouldshuffle = hparams.shouldshuffle

        print('Finished loading {} dataset... '.format(mode))

    def load_data(self, sbmt, mode):
        # Load and slice data
        sp_tmp = np.load(os.path.join(self.root_dir, sbmt[2]))
        f0_tmp = np.load(os.path.join(self.feat_dir, sbmt[2]))
        if self.mode == 'train':
            sp_tmp = sp_tmp[self.split:, :]
            f0_tmp = f0_tmp[self.split:]
        elif self.mode == 'test':
            sp_tmp = sp_tmp[:self.split, :]
            f0_tmp = f0_tmp[:self.split]
        else:
            raise ValueError
        uttrs = (sp_tmp, sbmt[1], f0_tmp)
        return uttrs

    def gen_sample_array(self):
        initial = tf.random.uniform(shape=(), minval=0, maxval=self.num_samples-hparams.batch_size, dtype=tf.int64)
        self.sample_idx_array = tf.range(hparams.batch_size, dtype=tf.int64) + initial
        if self.shouldshuffle:
            self.sample_idx_array = tf.random.shuffle(self.sample_idx_array)
        return self.sample_idx_array
    
    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


    def collate(self, batch):
        
        new_batch = []
        for token in batch:
            aa, b, c = token
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq+1, size=2) # 1.5s ~ 3s
            left = np.random.randint(0, len(aa)-len_crop[0], size=2)
            #pdb.set_trace()
            
            a = aa[left[0]:left[0]+len_crop[0], :]
            c = c[left[0]:left[0]+len_crop[0]]
            
            a = np.clip(a, 0, 1)
            
            a_pad = np.pad(a, ((0,self.max_len_pad-a.shape[0]),(0,0)), 'constant')
            c_pad = np.pad(c[:,np.newaxis], ((0,self.max_len_pad-c.shape[0]),(0,0)), 'constant', constant_values=-1e10)
            
            new_batch.append( (a_pad, b, c_pad, len_crop[0]) ) 
            
        batch = new_batch  
        

        a, b, c, d = zip(*batch)
        melsp = tf.convert_to_tensor(a, dtype=tf.float32)
        spk_emb = tf.convert_to_tensor(b, dtype=tf.float32)
        pitch = tf.convert_to_tensor(c, dtype=tf.float32)
        len_org = tf.convert_to_tensor(d, dtype=tf.int32)
        
        return melsp, spk_emb, pitch, len_org

    

def get_loader(mode):
    dataset = SpeechSplitDataset(hparams.root_dir, hparams.feat_dir, mode)
    return dataset
    
