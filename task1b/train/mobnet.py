"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""
import keras
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Input, concatenate, Lambda
from keras.layers import  Add, Reshape, DepthwiseConv2D, Dropout
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K

def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):

    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    return x


def pad_depth(inputs, desired_channels):
    from keras import backend as K
    y = K.zeros_like(inputs, name='pad_depth1')
    return y


def freq_split1(x):
    from keras import backend as K
    return x[:, 0:64, :, :]


def freq_split2(x):
    from keras import backend as K
    return x[:, 64:128, :, :]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def mobile_net_block(inputs, first_filters, alpha):
    x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))
    x = _inverted_residual_block(x, 32, (3, 3), t=2, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 40, (3, 3), t=2, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 48, (3, 3), t=2, alpha=alpha, strides=2, n=3)

    if alpha > 1.0:
        last_filters = _make_divisible(80 * alpha, 8)
    else:
    	last_filters = 56

    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))

    return x


def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation('relu')(x)


def model_mobnet(num_classes, input_shape=[128, None, 2], num_filters=24, wd=1e-3, alpha=1):

    num_res_blocks=2
    
    inputs = Input(shape=input_shape)
    
    #split up frequency into two branches
    Split1 =  Lambda(freq_split1)(inputs)
    Split2 =  Lambda(freq_split2)(inputs)
    first_filters = _make_divisible(32 * alpha, 8)

    Split1 = mobile_net_block(Split1, first_filters, alpha)
    Split2 = mobile_net_block(Split2, first_filters, alpha)


    MobilePath = concatenate([Split1, Split2],axis=1)

    OutputPath = resnet_layer(inputs=MobilePath,
                              num_filters=2*num_filters,
                              kernel_size=1,
                              strides=1,
                              learn_bn = False,
                              wd=wd,
                              use_relu = True)

    OutputPath = Dropout(0.3, name='Dropout')(OutputPath)

        
    OutputPath = resnet_layer(inputs=OutputPath,
                     num_filters=num_classes,
                     strides = 1,
                     kernel_size=1,
                     learn_bn = False,
                     wd=wd,
                     use_relu=False)

    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)

    model = Model(inputs=inputs, outputs=OutputPath)
    return model

model =  model_mobnet(3, input_shape=[128, 461, 3*2], num_filters=24, wd=1e-3)
model.load_weights("")
model.summary()
