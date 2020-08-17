
import keras
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Input, concatenate, Lambda
from keras.regularizers import l2
from keras.models import Model

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

def My_freq_split1(x):
    return x[:,0:64,:,:]

def My_freq_split2(x):
    return x[:,64:128,:,:]



def model_resnet(num_classes,input_shape =[128,None,1], num_filters =24,wd=1e-3):
    
    num_res_blocks=2
    inputs = Input(shape=input_shape)
    Split1=  Lambda(My_freq_split1)(inputs)
    Split2=  Lambda(My_freq_split2)(inputs)

    ResidualPath1 = resnet_layer(inputs=Split1,
                     num_filters=num_filters,
                     strides=[1,2],
                     learn_bn = True,
                     wd=wd,
                     use_relu = False)
    
    ResidualPath2 = resnet_layer(inputs=Split2,
                     num_filters=num_filters,
                     strides=[1,2],
                     learn_bn = True,
                     wd=wd,
                     use_relu = False)

    # Instantiate the stack of residual units
    for stack in range(4):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0: 
                strides = [1,2]  # downsample
            ConvPath1 = resnet_layer(inputs=ResidualPath1,
                             num_filters=num_filters,
                             strides=strides,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            ConvPath2 = resnet_layer(inputs=ResidualPath2,
                             num_filters=num_filters,
                             strides=strides,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            ConvPath1 = resnet_layer(inputs=ConvPath1,
                             num_filters=num_filters,
                             strides=1,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            ConvPath2 = resnet_layer(inputs=ConvPath2,
                             num_filters=num_filters,
                             strides=1,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
            if stack > 0 and res_block == 0:  
                # average pool and downsample the residual path
                ResidualPath1 = AveragePooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(ResidualPath1)
                ResidualPath2 = AveragePooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(ResidualPath2)
                
                # zero pad to increase channels
                desired_channels = ConvPath1.shape.as_list()[-1]

                Padding1=Lambda(pad_depth,arguments={'desired_channels':desired_channels})(ResidualPath1)
                ResidualPath1 = keras.layers.Concatenate(axis=-1)([ResidualPath1,Padding1])
                
                Padding2=Lambda(pad_depth,arguments={'desired_channels':desired_channels})(ResidualPath2)
                ResidualPath2 = keras.layers.Concatenate(axis=-1)([ResidualPath2,Padding2])

            ResidualPath1 = keras.layers.add([ConvPath1,ResidualPath1])
            ResidualPath2 = keras.layers.add([ConvPath2,ResidualPath2])
            
        num_filters *= 2
        

    ResidualPath = concatenate([ResidualPath1,ResidualPath2],axis=1)
    OutputPath = resnet_layer(inputs=ResidualPath,
                             num_filters=2*num_filters,
                              kernel_size=1,
                             strides=1,
                             learn_bn = False,
                             wd=wd,
                             use_relu = True)
    #output layers after last sum
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
