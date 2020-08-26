import librosa
import numpy as np
import tensorflow as tf


def getTrainableVariables(tag=""):
    return [v for v in tf.trainable_variables() if tag in v.name]


def getNumParams(tensors):
    return np.sum([np.prod(t.get_shape().as_list()) for t in tensors])


def crop_and_concat(x1, x2, level, match_feature_dim = True,  attention_mechanism = False, scalar_skip_connection = False):
    '''
    Copy-and-crop operation for two feature maps of different size.
    Crops the first input x1 equally along its borders so that its shape is equal to 
    the shape of the second input x2, then concatenates them along the feature channel axis.
    :param x1: First input that is cropped and combined with the second input
    :param x2: Second input
    :return: Combined feature map
    '''
    if x2 is None:
        return x1
    
    x1 = crop(x1,x2.get_shape().as_list(), match_feature_dim)
    
    if attention_mechanism == True:
        u = 1
        Wx = tf.layers.conv1d(x1, u, 1)
        Wg = tf.layers.conv1d(x2, u, 1)
        B = tf.sigmoid(Wx + Wg)
        Wf = tf.layers.conv1d(B, 1, 1)
        A = tf.sigmoid(Wf)
        x1 = tf.multiply(A, x1)

    if scalar_skip_connection == True:
        shape = np.array(x1.get_shape().as_list())
        weights = tf.get_variable("scalar_" + str(level), initializer = np.ones(shape[2], dtype=np.float32), dtype=tf.float32)
        x1 = tf.multiply(x1, weights)
        
    return tf.concat([x1, x2], axis=2)


def concat_z(x):
  shape = x.get_shape().as_list() 
  z = tf.random_normal((1,shape[1],1), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
  z = tf.Variable(z, trainable=False)

  dynamic_shape = tf.broadcast_dynamic_shape( (1,shape[1],1), (shape[0],shape[1],1) )
  return tf.concat([x, tf.tile(z, [dynamic_shape[0],1,1])], axis=2)


def crop_sample(sample, crop_frames, input):
    for key, val in list(sample.items()):
        if key != input and crop_frames > 0: # if key is different from the input key
            sample[key] = val[crop_frames:-crop_frames,:]
    return sample


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def AudioClip(x, training):
    '''
    Simply returns the input if training is set to True, otherwise clips the input to [-1,1]
    :param x: Input tensor (coming from last layer of neural network)
    :param training: Whether model is in training (True) or testing mode (False)
    :return: Output tensor (potentially clipped)
    '''
    if training:
        return x
    else:
        return tf.maximum(tf.minimum(x, 1.0), -1.0)


def resample(audio, orig_sr, new_sr):
    if orig_sr == new_sr: # if orig_sr and new_sr are different
        return audio
    return librosa.resample(audio.T, orig_sr, new_sr).T


def load(path, sr=None, mono=True, offset=0.0, duration=None, dtype=np.float32):
    # ALWAYS output (n_frames, n_channels) audio
    y, orig_sr = librosa.load(path, sr, mono, offset, duration, dtype)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=0)
    return y.T, orig_sr


def crop(tensor, target_shape, match_feature_dim=True):
    '''
    Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
    Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
    :param tensor: 3D tensor [batch_size, width, channels] that should be cropped. 
    :param target_shape: Target shape (3D tensor) that the tensor should be cropped to
    :return: Cropped tensor
    '''
    shape = np.array(tensor.get_shape().as_list())
    diff = shape - np.array(target_shape)
    assert(diff[0] == 0 and (diff[2] == 0 or not match_feature_dim)) # Only width axis can differ
    if (diff[1] % 2 != 0):
        print("WARNING: Cropping with uneven number of extra entries on one side")
    assert diff[1] >= 0 # Only positive difference allowed
    if diff[1] == 0:
        return tensor
    crop_start = diff // 2
    crop_end = diff - crop_start

    return tensor[:,crop_start[1]:-crop_end[1],:]


def stft_loss(x, y, sample_rate=44100, frame_length=512):
    X_mag_spec = get_mag_spec(x, sample_rate, frame_length)
    Y_mag_spec = get_mag_spec(y, sample_rate, frame_length)
    return tf.abs(Y_mag_spec - X_mag_spec)

def get_mag_spec(inputs, sample_rate=44100, frame_length=512):
    spec = tf.contrib.signal.stft(tf.squeeze(inputs, 2), frame_length=frame_length, frame_step=frame_length//2, fft_length=int(2**np.ceil(np.log2(frame_length))))
    return tf.abs(spec)


def perceptual_loss(x, y, sample_rate=44100, frame_length=1024):
    X_mel = get_mel_spec(x, sample_rate, frame_length)
    Y_mel = get_mel_spec(y, sample_rate, frame_length)
    return tf.abs(Y_mel - X_mel)

def get_mel_spec(inputs, sample_rate=44100, frame_length=1024):
    spec = tf.contrib.signal.stft(tf.squeeze(inputs, 2), frame_length=frame_length, frame_step=frame_length//2, fft_length=int(2**np.ceil(np.log2(frame_length))))
    mag_spec = tf.abs(spec)
    n_bins = frame_length//2 + 1
    lower_hz, high_hz, n_mel_bins = 20.0, 22050.0, frame_length//2 + 1
    l2mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(n_mel_bins, n_bins, sample_rate, lower_hz, high_hz)
    mel_spec = tf.tensordot(mag_spec, l2mel_matrix, 1)
    return mel_spec