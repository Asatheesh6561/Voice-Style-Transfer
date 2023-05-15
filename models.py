import tensorflow as tf
import hparams
class LinearNorm(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(out_dim, use_bias=bias, kernel_initializer='GlorotNormal')

    def call(self, x):
        return self.linear_layer(x)
class GroupNorm(tf.keras.layers.Layer):
    def __init__(self, group, axis=-1, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.group = group
        self.axis = axis
        self.eps = eps
    
    def build(self, input_shape):
        C = input_shape[self.axis]
        self.gamma = self.add_weight(shape=(C,), name='gamma', initializer='ones')
        self.beta = self.add_weight(shape=(C,), name='beta', initializer='zeros')
        self.built = True
    
    def call(self, inputs):
        N, C, T = inputs.shape
        group_shape = (N, self.group, T)
        x = tf.reshape(inputs, group_shape + (C // self.group,))
        mean, var = tf.nn.moments(x, [1, 2, 3], keepdims=True)

        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, (N, C, T))

        return x * self.gamma + self.beta

class ConvNorm(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding='same', dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        self.conv = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, 
                                           strides=stride, padding='same',
                                           dilation_rate=dilation, use_bias=bias)

        gain = 2**0.5 if w_init_gain == 'relu' else 1
        self.conv.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=gain)

    def call(self, signal):
        signal = tf.transpose(signal, perm = [0, 2, 1])
        conv_signal = self.conv(signal)
        return tf.transpose(conv_signal, perm = [0, 2, 1])
class Encoder_t(tf.keras.Model):
    """Rhythm Encoder"""
    def __init__(self):
        super(Encoder_t, self).__init__()

        self.dim_neck_2 = hparams.dim_neck_2
        self.freq_2 = hparams.freq_2
        self.dim_freq = hparams.dim_freq
        self.dim_enc_2 = hparams.dim_enc_2
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp

        convolutions = []
        for i in range(1):
            conv_layer = tf.keras.Sequential([
                ConvNorm(self.dim_freq if i == 0 else self.dim_enc_2,
                         self.dim_enc_2, 
                         kernel_size=5, stride=1,
                         padding=2, 
                         dilation=1, w_init_gain='relu'),
                GroupNorm(self.dim_enc_2//self.chs_grp)])
            convolutions.append(conv_layer)
        self.convolutions = convolutions
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))

    def call(self, x, mask):
        for conv in self.convolutions:
            x = conv(x)
            x = tf.nn.relu(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        outputs = self.lstm(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2:]

        codes = tf.concat((out_forward[:,self.freq_2-1::self.freq_2,:], out_backward[:,::self.freq_2,:]), axis=-1)
        return codes

class Encoder_6(tf.keras.Model):
    """F0 Encoder"""
    def __init__(self):
        super(Encoder_6, self).__init__()

        self.dim_neck_3 = hparams.dim_neck_3
        self.freq_3 = hparams.freq_3
        self.dim_f0 = hparams.dim_f0
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp
        self.len_org = tf.constant(hparams.max_len_pad, shape=[])

        convolutions = []
        for i in range(3):
            conv_layer = tf.keras.Sequential([
                ConvNorm(self.dim_f0 if i==0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                GroupNorm(self.dim_enc_3//self.chs_grp)
            ])
            convolutions.append(conv_layer)
        self.convolutions = convolutions
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.dim_neck_3, return_sequences=True))
        self.interp = InterpLnr()

    def call(self, x):
        for conv in self.convolutions:
            x = tf.nn.relu(conv(x))
            x = tf.transpose(x, perm=[1, 2])
            x = self.interp(x, tf.expand_dims(self.len_org, axis=0))
            x = tf.transpose(x, perm=[1, 2])
        x = tf.transpose(x, perm=[1, 2])
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck_3]
        out_backward = outputs[:, :, self.dim_neck_3:]
        codes = tf.concat((out_forward[:, self.freq_3-1::self.freq_3, :], out_backward[:, ::self.freq_3, :]), axis=-1)
        return codes

class Encoder_7(tf.keras.Model):
    """Sync Encoder"""
    def __init__(self):
        super(Encoder_7, self).__init__()
        self.dim_neck = hparams.dim_neck
        self.freq = hparams.freq
        self.freq_3 = hparams.freq_3
        self.dim_enc = hparams.dim_enc
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_freq = hparams.dim_freq
        self.chs_grp = hparams.chs_grp
        self.len_org = tf.constant(hparams.max_len_pad, shape=[])
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_f0 = hparams.dim_f0

        #convolutions for code1
        convolutions = []
        for i in range(3):
            conv_layer = tf.keras.Sequential([
                ConvNorm(self.dim_freq if i==0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                GroupNorm(self.dim_enc//self.chs_grp)
            ])
            convolutions.append(conv_layer)
        self.convolutions_1 = convolutions
        self.lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.dim_neck, return_sequences=True))

        #convolutions for f0
        convolutions = []
        for i in range(3):
            conv_layer = tf.keras.Sequential([
                ConvNorm(self.dim_f0 if i==0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                GroupNorm(self.dim_enc_3//self.chs_grp)
            ])
            convolutions.append(conv_layer)
        self.convolutions_2 = convolutions
        self.lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.dim_neck_3, return_sequences=True))
        self.interp = InterpLnr()

    def call(self, x_f0):
        x = x_f0[:, :self.dim_freq, :]
        f0 = x_f0[:, self.dim_freq:, :]

        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            x = conv_1(x)
            x = tf.nn.relu(x)
            f0 = conv_2(f0)
            f0 = tf.nn.relu(f0)
            x_f0 = tf.transpose(tf.concat((x, f0), axis=1), perm=[0, 2, 1])
            x_f0 = self.interp(x_f0, self.len_org * tf.ones([x.shape[0]], dtype=self.len_org.dtype))
            x_f0 = tf.transpose(x_f0, [0, 2, 1])
            x = x_f0[:, :self.dim_enc, :]
            f0 = x_f0[:, self.dim_enc:, :]

        x_f0 = tf.transpose(x_f0, [0, 2, 1])    
        x = x_f0[:, :, :self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc:]

        # code 1
        x = self.lstm_1(x)
        f0 = self.lstm_2(f0)

        x_forward = x[:, :, :self.dim_neck]
        x_backward = x[:, :, self.dim_neck:]

        f0_forward = f0[:, :, :self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3:]

        codes_x = tf.concat((x_forward[:,self.freq-1::self.freq,:], 
                            x_backward[:,::self.freq,:]), axis=-1)

        codes_f0 = tf.concat((f0_forward[:,self.freq_3-1::self.freq_3,:], 
                            f0_backward[:,::self.freq_3,:]), axis=-1)

        return codes_x, codes_f0 

class Decoder_3(tf.keras.Model):
    """Decoder Module"""
    def __init__(self):
        super(Decoder_3, self).__init__()
        self.dim_neck = hparams.dim_neck
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_spk_emb = hparams.dim_spk_emb
        self.dim_freq = hparams.dim_freq
        self.dim_neck_3 = hparams.dim_neck_3

        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True, kernel_initializer='glorot_uniform'))
        self.linear_projection = LinearNorm(512, self.dim_freq)

    def call(self, x):
        outputs = self.lstm(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output
    
class Decoder_4(tf.keras.Model):
    """F0 Converter"""
    def __init__(self):
        super(Decoder_4, self).__init__()
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_f0 = hparams.dim_f0
        self.dim_neck_3 = hparams.dim_neck_3

        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(None, self.dim_neck_2*2+self.dim_neck_3*2)))
        self.linear_projection = LinearNorm(256, self.dim_f0)
    
    def call(self, x):
        outputs = self.lstm(x)
        decoder_outputs = self.linear_projection(outputs)
        return decoder_outputs

class Generator_3(tf.keras.Model):
    """SpeechSplit Model"""
    def __init__(self):
        super(Generator_3, self).__init__()
        self.encoder1 = Encoder_7()
        self.encoder2 = Encoder_t()
        self.decoder = Decoder_3()

        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def call(self, x_f0, x_org, c_trg):
        x_1 = tf.transpose(x_f0, perm=[0, 2,1])
        codes_x, codes_f0 = self.encoder1(x_1)
        code_exp_1 = tf.repeat(codes_x, self.freq, axis=1)
        code_exp_3 = tf.repeat(codes_f0, self.freq_3, axis=1)
        
        x_2 = tf.transpose(x_org, perm=[0, 2, 1])
        codes_2 = self.encoder2(x_2, None)
        code_exp_2 = tf.repeat(codes_2, self.freq_2, axis=1)
        
        encoder_outputs = tf.concat([code_exp_1, code_exp_2, code_exp_3, tf.ones((c_trg.shape[0], x_1.shape[-1], c_trg.shape[-1]))*tf.expand_dims(c_trg, axis=1)], axis=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs

    def rhythm(self, x_org):
        x_2 = tf.transpose(x_org, perm=[2,1])
        codes_2 = self.encoder_2(x_2, None)
        
        return codes_2

class Generator_6(tf.keras.Model):
    """F0 Converter"""
    def __init__(self):
        super(Generator_6).__init__()
        self.encoder_2 = Encoder_t(hparams)
        self.encoder_3 = Encoder_6(hparams)
        self.decoder = Decoder_4(hparams)
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def call(self, x_org, f0_trg):
        x_2 = tf.transpose(x_org, perm=[2,1])
        codes_2 = self.encoder_2(x_2, None)
        code_exp_2 = tf.repeat(codes_2, self.freq_2, axis=1)
        
        x_3 = tf.transpose(f0_trg, perm=[2,1])
        codes_3 = self.encoder_3(x_3)
        code_exp_3 = tf.repeat(codes_3, self.freq_3, axis=1)
        
        encoder_outputs = tf.concat([code_exp_2, code_exp_3], axis=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs

        
class InterpLnr(tf.keras.Model):
    def __init__(self):
        super(InterpLnr, self).__init__()
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad
        
        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg
        
        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1
    
    # def pad_sequences(self, sequences):
    #     # Get the channel dimension of the first sequence
    #     channel_dim = tf.shape(sequences[0])[-1]

    #     # Create an empty tensor with the same dimensions as the sequences
    #     out_dims = (len(sequences), self.max_len_pad, channel_dim)
    #     out_tensor = tf.zeros(out_dims, dtype=sequences[0].dtype)

    #     # Iterate through the sequences and update the out_tensor
    #     for i, tensor in enumerate(sequences):
    #         length = tf.shape(tensor)[0]
    #         indices = tf.stack([tf.ones(length, dtype=tf.int32)*i, tf.range(length)], axis=1)
    #         out_tensor = tf.tensor_scatter_nd_update(out_tensor, indices, tensor[:self.max_len_pad])

    #     return out_tensor
    def pad_sequences(self, sequences):
        padded_sequences = []
        for tensor in sequences:
            length = tf.shape(tensor)[0]
            if length <= self.max_len_pad: padded_tensor = tf.pad(tensor, [[0, self.max_len_pad - length], [0, 0]], constant_values=0)
            else: padded_tensor = tensor[:self.max_len_pad]
            padded_sequences.append(padded_tensor)
        out_tensor = tf.stack(padded_sequences, axis=0)
        return out_tensor

    def call(self, x, len_seq):
        # if not self.training:
        #     return x
        batch_size = x.shape[0]
        indices = tf.range(self.max_len_seg*2)
        indices = tf.expand_dims(indices, 0)
        indices = tf.tile(indices, [batch_size*self.max_num_seg, 1])
        scales = tf.random.uniform(shape=[batch_size*self.max_num_seg,], minval=0.5, maxval=1.5, dtype=tf.float32)
        
        idx_scaled = tf.cast(indices, dtype=tf.float32) / tf.expand_dims(scales, -1)
        idx_scaled_fl = tf.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        thing = tf.cast(tf.linspace(self.min_len_seg, self.max_len_seg, batch_size*self.max_num_seg//2), dtype=tf.int32)
        len_seg = tf.random.uniform(shape=[batch_size*self.max_num_seg,1], minval=self.min_len_seg, maxval=self.max_len_seg, dtype=tf.int32)
        
        idx_mask = idx_scaled_fl < tf.cast(len_seg - 1, dtype=tf.float32)

        offset = tf.cumsum(tf.reshape(len_seg, (batch_size, -1)), axis=-1)
        offset = tf.slice(offset, [0, 0], [-1, -1])
        paddings = [[0, 0], [1, 0]]
        offset = tf.pad(offset, paddings, constant_values=0)[:, :-1]
        offset = tf.reshape(offset, [-1, 1])
        idx_scaled_org = idx_scaled_fl + tf.cast(offset, dtype=tf.float32)
        len_seq_rp = tf.repeat(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < tf.cast((len_seq_rp - 1)[:, tf.newaxis], dtype=tf.float32)
        idx_mask_final = idx_mask & idx_mask_org
        counts = tf.reduce_sum(tf.cast(idx_mask_final, tf.int32), axis=-1)
        counts = tf.reduce_sum(tf.reshape(counts, (batch_size, -1)), axis=-1)

        index_1 = tf.repeat(tf.range(batch_size, dtype=tf.int64), counts)
        index_2_fl = tf.cast(idx_scaled_org[idx_mask_final], tf.int64)
        index_2_cl = index_2_fl + 1

        y_fl = tf.gather_nd(x, tf.stack([index_1, index_2_fl], axis=-1))
        y_cl = tf.gather_nd(x, tf.stack([index_1, index_2_cl], axis=-1))
        lambda_f = lambda_[idx_mask_final][:, tf.newaxis]

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl

        sequences = tf.split(y, counts.numpy().tolist(), axis=0)
        seq_padded = self.pad_sequences(sequences)
        return seq_padded 

    

