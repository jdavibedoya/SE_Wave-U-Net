import Models.InterpolationLayer
import Utils
from Utils import LeakyReLU

import numpy as np
import tensorflow as tf


class UnetAudioEnhancer:
    '''
    U-Net network for audio enhancement.
    Uses valid convolutions, so it predicts for the centre part of the input - only certain input and output shapes are therefore possible (see get_padding function)
    '''

    def __init__(self, model_config):
        '''
        Initialize U-Net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.filter_size = model_config["filter_size"]
        self.merge_filter_size = model_config["merge_filter_size"]
        self.input_filter_size = model_config["input_filter_size"]
        self.output_filter_size = model_config["output_filter_size"]
        self.upsampling = model_config["upsampling"]
        self.padding = model_config["padding"]
        self.num_channels = model_config["num_channels"]
        self.output_activation = model_config["output_activation"]
        self.noise_input_vector = model_config["noise_input_vector"]
        self.attention_mechanism = model_config["attention_mechanism"]
        self.scalar_skip_connection = model_config["scalar_skip_connection"]
        self.skipping_post_activation = model_config["skipping_post_activation"]
        self.z_latent = model_config["z_latent"]

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the U-Net works and has the given shape as output shape
        :param shape: Desired output shape 
        :return: input_shape, output_shape, where each is a list [batch_size, time_steps, channels]
        '''

        # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
        rem = float(shape[1]) # Cut off batch size number and channel

        # Output filter size
        rem = rem + self.output_filter_size - 1

        # Upsampling blocks
        for i in range(self.num_layers):
            rem = rem + self.merge_filter_size - 1
            rem = (rem + 1.) / 2. # out = in + in - 1 <=> in = (out+1)/2

        # Round resulting feature map dimensions up to nearest integer
        x = np.asarray(np.ceil(rem),dtype=np.int64)
        assert(x >= 2)

        # Compute input and output shapes based on lowest-res feature map
        output_shape = x
        input_shape = x

        # Extra conv
        input_shape = input_shape + self.filter_size - 1

        # Go from centre feature map through upsampling and downsampling blocks
        for i in range(self.num_layers):
            output_shape = 2*output_shape - 1 # Upsampling
            output_shape = output_shape - self.merge_filter_size + 1 # Conv

            input_shape = 2*input_shape - 1 # Decimation
            if i < self.num_layers - 1:
                input_shape = input_shape + self.filter_size - 1 # Conv
            else:
                input_shape = input_shape + self.input_filter_size - 1

        # Output filters
        output_shape = output_shape - self.output_filter_size + 1

        input_shape = np.concatenate([[shape[0]], [input_shape], [self.num_channels]])
        output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])

        return input_shape, output_shape

    def get_output(self, input, training, reuse=True):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch, 3D tensor [batch_size, num_samples, num_channels]
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :return: U-Net output: List of estimates. Each item is a 3D tensor [batch_size, num_out_samples, num_channels]
        '''
        with tf.variable_scope("enhancer", reuse=reuse):
            enc_outputs = list()
            current_layer = input
            
            if self.noise_input_vector:
              current_layer = Utils.concat_z(current_layer)

            # Down-convolution: Repeat strided conv
            for i in range(self.num_layers):
                if self.skipping_post_activation == True:
                    current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * i), self.filter_size, strides=1, activation=LeakyReLU, padding=self.padding) # out = in - filter + 1
                    enc_outputs.append(current_layer)
                else:
                    current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * i), self.filter_size, strides=1, padding=self.padding) # out = in - filter + 1
                    enc_outputs.append(current_layer)
                    current_layer = LeakyReLU(current_layer)
                current_layer = current_layer[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1

            current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * self.num_layers), self.filter_size, activation=LeakyReLU, padding=self.padding) # One more conv here since we need to compute features after last decimation
            
            if self.z_latent:
              current_layer = Utils.concat_z(current_layer)

            # Feature map here shall be X along one dimension

            # Upconvolution
            for i in range(self.num_layers):
                #UPSAMPLING
                current_layer = tf.expand_dims(current_layer, axis=1)
                if self.upsampling == 'learned':
                    # Learned interpolation between two neighbouring time positions by using a convolution filter of width 2, and inserting the responses in the middle of the two respective inputs
                    current_layer = Models.InterpolationLayer.learned_interpolation_layer(current_layer, self.padding, i)
                else:
                    current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)

                current_layer = tf.squeeze(current_layer, axis=1)
                # UPSAMPLING FINISHED
                
                # Cropping as we are using context
                current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, i, match_feature_dim = False, attention_mechanism = self.attention_mechanism, scalar_skip_connection = self.scalar_skip_connection)
                current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * (self.num_layers - i - 1)), self.merge_filter_size, activation=LeakyReLU, padding=self.padding)  # out = in - filter + 1
            
            current_layer = Utils.crop_and_concat(input, current_layer, self.num_layers, match_feature_dim = False, attention_mechanism = self.attention_mechanism, scalar_skip_connection = self.scalar_skip_connection)

            # Output layer
            # Determine output activation function
            if self.output_activation == "tanh":
                out_activation = tf.tanh
            elif self.output_activation == "linear":
                print("Linear Output :)")
                out_activation = lambda x: Utils.AudioClip(x, training)
            else:
                raise NotImplementedError

            return tf.layers.conv1d(current_layer, self.num_channels, self.output_filter_size, activation=out_activation, padding=self.padding)