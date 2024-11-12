import tensorflow as tf
from tensorflow.keras import layers, Model

def build_unet(input_shape=(64, 64, 8)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    conv1 = create_encoder_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = create_encoder_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = create_encoder_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = create_bridge_block(pool3, 512)
    
    # Decoder
    up1 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    conv5 = create_decoder_block(up1, conv3, 256)
    
    up2 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    conv6 = create_decoder_block(up2, conv2, 128)
    
    up3 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    conv7 = create_decoder_block(up3, conv1, 64)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    
    return Model(inputs=inputs, outputs=outputs)

def create_encoder_block(inputs, filters):
    conv = layers.Conv2D(filters, 3, padding='same', activation='relu')(inputs)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(filters, 3, padding='same', activation='relu')(conv)
    return layers.BatchNormalization()(conv)

def create_bridge_block(inputs, filters):
    conv = layers.Conv2D(filters, 3, padding='same', activation='relu')(inputs)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(filters, 3, padding='same', activation='relu')(conv)
    conv = layers.BatchNormalization()(conv)
    return layers.Dropout(0.5)(conv)

def create_decoder_block(inputs, skip_connection, filters):
    concat = layers.Concatenate()([skip_connection, inputs])
    conv = layers.Conv2D(filters, 3, padding='same', activation='relu')(concat)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Conv2D(filters, 3, padding='same', activation='relu')(conv)
    return layers.BatchNormalization()(conv)
