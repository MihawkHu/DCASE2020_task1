from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda, Layer, Dot
from keras import backend as K
from keras.activations import sigmoid, softmax
from keras import initializers
import numpy as np
import tensorflow as tf

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature) #[None, 128, 29, 1]
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature) #[None, 128, 29, 1]
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool]) #[None, 128, 29, 2]
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature._keras_shape[-1] == 1 #[None, 128, 29, 1]
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])



#attention block
class my_attention(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(my_attention, self).__init__(**kwargs)

    # def build(self, input_shape):
        #为该层创建一个可训练的权重
                # 创建一个可训练的权重变量矩阵
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer=initializers.Orthogonal(gain=1.0, seed=None),
                                      trainable=True)
        # super(my_attention, self).build(input_shape)

    def call(self, x): #(None, 29, 768)
        s = K.mean(x, axis=1, keepdims=False) # batchsize channel(None, 768)
        att_weight1 = K.softmax(K.dot(s, self.kernel), axis=-1) # batchsize class (None, 10)
        C_1 = K.dot(att_weight1, K.transpose(self.kernel)) # batchsize channel (None, 768)
        C_1 = K.expand_dims(C_1, axis=-1) # batchsize channel *1 (None, 768, 1)

        att_weight2 = K.softmax(K.batch_dot(C_1, x, axes=[1, 2]), axis=-1) # batch * 1 * time (None, 1, 29)
        s_1 = K.batch_dot(att_weight2, x, axes=[2,1]) # batch * 1 * channel (None, 1, 768)
        s_1 = K.squeeze(s_1, axis=1) # batch channel (None, 768)
        # print(K.int_shape(s_1))
        # s_1 = K.expand_dims(s_1, axis=1)
        # print(K.is_keras_tensor(x))
        # print(K.int_shape(s_1))
        # s_1 = np.asarray([1,2,3])
        #s_1 = tf.covert_to_tensor(s_1)
        return s_1

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

def secondary_attention(input_feature): #(None, 128, 29, 768)
    if K.image_data_format() == 'channels_last':
        freq_axis = 1
        time_axis = 2
        channel_axis = 3

    b, freq, time, channel = input_feature._keras_shape
    x = AveragePooling2D((freq,1), strides=(freq,1))(input_feature) #(None, 1, 29, 768) batchsize * 1 * time * channel
    x = Lambda(lambda z: K.squeeze(z, axis=1))(x) #(None, 29, 768) batchsize * time * channel
    channel = x._keras_shape[-1]
    att = my_attention(channel, 10) # batchsize channel (None, 768)
    s_1 = Lambda(att.call)(x)
    # print(K.int_shape(s_1))
    return s_1


