from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Lambda, Concatenate, Add, Multiply

def res2net_bottleneck_block(x, f, s=4, expansion=4, use_se_block=False):
    """
    Arguments:
        x: input tensor
        f: number of output  channels
        s: scale dimension
    """
    
    num_channels = int(x._keras_shape[-1])
    assert num_channels % s == 0, f"Number of input channel should be a multiple of s. Received nc={num_channels} and s={s}."
    
    input_tensor = x
    
    # Conv 1x1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f, 1, kernel_initializer='he_normal', use_bias=False)(x)
    
    # Conv 3x3
    subset_x = []
    n = f
    w = n // s
    for i in range(s):
        slice_x = Lambda(lambda x: x[..., i*w:(i+1)*w])(x)
        if i > 1:
            slice_x = Add()([slice_x, subset_x[-1]])
        if i > 0:
            slice_x = BatchNormalization()(slice_x)
            slice_x = Activation('relu')(slice_x)
            slice_x = Conv2D(w, 3, kernel_initializer='he_normal', padding='same', use_bias=False)(slice_x)        
        subset_x.append(slice_x)
    x = Concatenate()(subset_x)
        
    # Conv 1x1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(f*expansion, 1, kernel_initializer='he_normal', use_bias=False)(x)
    
    if use_se_block:
        x = se_block(x)
    
    # Add
    if num_channels == f*expansion:
        skip = input_tensor
    else:
        skip = input_tensor
        skip = Conv2D(f*expansion, 1, kernel_initializer='he_normal')(skip)
    out = Add()([x, skip])
    return out

def se_block(input_tensor, c=16):
    num_channels = int(input_tensor._keras_shape[-1]) # Tensorflow backend
    bottleneck = int(num_channels // c)
 
    se_branch = GlobalAveragePooling2D()(input_tensor)
    se_branch = Dense(bottleneck, use_bias=False, activation='relu')(se_branch)
    se_branch = Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)
    
    out = Multiply()([input_tensor, se_branch]) 
    return out