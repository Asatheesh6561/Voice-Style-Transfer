import tensorflow as tf
from models import Generator_3 as Generator
from models import InterpLnr
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import pickle
import hparams
import logging
from utils import pad_seq_to_2, quantize_f0_tf
from data_loader import get_loader

validation_pt = pickle.load(open('assets/demo.pkl', "rb"))
class Solver():
    def __init__(self, dataset, config):
        #Dataset and Config
        self.dataset = dataset
        self.config = config

        #Training Configuration
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = tf.test.is_built_with_cuda()
        self.device = tf.device('gpu:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('run/logs/log.log')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        self.build_model()
    def build_model(self):
        self.G = Generator()
        self.Interp = InterpLnr()
        self.g_optimizer = tf.optimizers.Adam(learning_rate=self.g_lr, beta_1=self.beta1, beta_2=self.beta2)

    def print_network(self, model, name):
        """Print out the network information."""
        def summary_plus(layer, i=0):
            if hasattr(layer, 'layers'):
                if i != 0: 
                    layer.summary()
                for l in layer.layers:
                    i += 1
                    summary_plus(l, i=i)
        print(model)
        print(name)
        
        for i in range(len(model.layers)):
            print(model.layers[i].summary())
        
    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.h5'.format(resume_iters))
        # latest_checkpoint = tf.train.latest_checkpoint(self.model_save_dir)
        # if latest_checkpoint: tf.train.Checkpoint(model=self.G).restore(latest_checkpoint)
        test_outputs = self.G(tf.zeros([16, 192, 337]), tf.zeros([16, 192, 80]), tf.zeros([16, 82])) #hack for initializing h5 saved model files
        self.G.load_weights(G_path)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        tf.keras.backend.clear_session()

    def train(self):
        start_iters = 0
        if self.resume_iters:
            print('Resuming...')
            start_iters = self.resume_iters
            self.restore_model(start_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')
        g_lr = self.g_lr
        print ('Current learning rates, g_lr: {}.'.format(g_lr))
        
        keys = ['G/loss_id']
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            sampler = self.dataset.gen_sample_array()
            batch = [self.dataset.train_dataset[i] for i in (sampler)]
            x_real_org, emb_org, f0_org, len_org = self.dataset.collate(batch)
            with tf.GradientTape() as tape:
                x_f0 = tf.concat((x_real_org, f0_org), axis=-1)
                x_f0_intrp = self.Interp(x_f0, len_org)
                f0_org_intrp = quantize_f0_tf(x_f0_intrp[:, :, -1])[0]
                x_f0_intrp_org = tf.concat((x_f0_intrp[:,:,:-1], tf.reshape(f0_org_intrp, (hparams.batch_size, hparams.max_len_pad, -1))), axis=-1)
                x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
                mse = tf.keras.losses.MeanSquaredError()
                g_loss_id = tf.reduce_mean(mse(x_real_org, x_identic))

            
                # Backward and optimize
            g_loss = g_loss_id
            grads = tape.gradient(g_loss, self.G.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.G.trainable_weights))
            loss = {}
            loss['G/loss_id'] = g_loss
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)
                self.logger.info(log)
                

            if self.use_tensorboard:
                for tag, value in loss.items():
                    self.writer.add_scalar(tag, value, i+1)
                
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.h5'.format(i+1))
                self.G.save_weights(G_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            if (i + 1) % self.sample_step == 0:
                with tf.GradientTape() as tape:
                    val_loss = 0
                    for val_sub in validation_pt:
                        emb_org_val = tf.constant(val_sub[1], dtype=tf.float32)
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis, :, :], 192)
                            len_org = tf.constant([val_sub[k][2]], dtype=tf.int32)
                            f0_org = np.pad(val_sub[k][1], (0, 192-val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_tf(tf.convert_to_tensor(f0_org))[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = tf.constant(f0_onehot, dtype=tf.float32)
                            x_real_pad = tf.constant(x_real_pad, dtype=tf.float32)
                            x_f0 = tf.concat([x_real_pad, f0_org_val], axis=-1)
                            x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            mse = tf.keras.losses.MeanSquaredError()
                            g_loss_val = mse(x_identic_val, x_real_pad)
                            val_loss += g_loss_val
                            print(g_loss_val, val_loss)
                    val_loss = val_loss / len(validation_pt)
                    print('Validation loss: {}'.format(val_loss.numpy()))
                
            if (i+1) % self.sample_step == 0:
                with tf.GradientTape() as tape:
                    for val_sub in validation_pt:
                        x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis,:,:], 192)
                        len_org = tf.convert_to_tensor([val_sub[k][2]], dtype=tf.int32)
                        f0_org = np.pad(val_sub[k][1], (0, 192-val_sub[k][2]), 'constant', constant_values=(0, 0))
                        f0_quantized = quantize_f0_tf(tf.convert_to_tensor(f0_org))[0]
                        f0_org_val = tf.convert_to_tensor(f0_onehot, dtype=tf.float32)
                        x_real_pad = tf.convert_to_tensor(x_real_pad, dtype=tf.float32)
                        x_f0 = tf.concat([x_real_pad, f0_org_val], axis=-1)
                        x_f0_F = tf.concat([x_real_pad, tf.zeros_like(f0_org_val)], axis=-1)
                        x_f0_C = tf.concat([tf.zeros_like(x_real_pad), f0_org_val], axis=-1)

                        x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                        x_identic_woF = self.G(x_f0_F, x_real_pad, emb_org_val)
                        x_identic_woR = self.G(x_f0, tf.zeros_like(x_real_pad), emb_org_val)
                        x_identic_woC = self.G(x_f0_C, x_real_pad, emb_org_val)

                        melsp_gd_pad = x_real_pad[0].cpu().numpy().T
                        melsp_out = x_identic_val[0].cpu().numpy().T
                        melsp_woF = x_identic_woF[0].cpu().numpy().T
                        melsp_woR = x_identic_woR[0].cpu().numpy().T
                        melsp_woC = x_identic_woC[0].cpu().numpy().T

                        min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                        max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))

                        fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1, sharex=True)
                        im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                        im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                        im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                        im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                        im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                        plt.savefig(f'{self.sample_dir}/{i+1}_{val_sub[0]}_{k}.png', dpi=150)
                        plt.close(fig) 
                        

            
