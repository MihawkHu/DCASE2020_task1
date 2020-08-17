import keras
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, MaxPooling2D, Dense
from keras.layers import Input, Dropout, ZeroPadding2D
from keras.regularizers import l2
from keras.models import Model
from attention_layer import spatial_attention, channel_attention, cbam_block, secondary_attention
from keras import backend as K


def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):
    x = inputs
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='valid',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    return x

def conv_layer1(inputs, num_channels=6, num_filters=14, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size1 = [5, 5]
    kernel_size2 = [3, 3]
    strides1 = [1, 2]
    strides2 = [1, 1]
    num_channels = 1

    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)

    # 1 
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size1, strides=strides1,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    # 2
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size2, strides=strides2,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3),strides=[2,2], padding='same')(x)
    return x


def conv_layer2(inputs, num_channels=6, num_filters=28, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    num_channels = 1

    x = inputs

    #1
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    #2
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)


    x = MaxPooling2D(pool_size=(3, 3), strides=[2,2], padding='same')(x)
    return x

def conv_layer3(inputs, num_channels=6, num_filters=56, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    num_channels = 1


    x = inputs
    # 1
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    # x = Dropout(0.2)(x)
    #2
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
 
    # Max Pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(x) # with [2, 2] and all data i achived 71.2 but 67 on s4-s6

    # 3
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    # x = Dropout(0.2)(x)
    #4
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(x)
    return x

def conv_layer4(inputs, num_channels=6, num_filters=128, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    num_channels = 1

    x = inputs

    # 1
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # 2
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    return x




def model_fcnn(num_classes, input_shape=[128, None, 3], num_filters=[14, 24, 48, 96], wd=1e-3):

    inputs = Input(shape=input_shape)

    
    ConvPath1 = conv_layer1(inputs=inputs,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    ConvPath4 = conv_layer4(inputs=ConvPath3,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[3],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    # output layers after last sum
    OutputPath = resnet_layer(inputs=ConvPath4,
                              num_filters=num_classes,
                              strides=1,
                              kernel_size=1,
                              learn_bn=False,
                              wd=wd,
                              use_relu=True)

    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = channel_attention(OutputPath, ratio=2)

    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model

