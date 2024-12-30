import tensorflow as tf
from tensorflow.keras import layers, models 

def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    #encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    #bottel
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    #decoder
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(x)  
    model = models.Model(inputs, outputs)

    return model

#usage
model = create_model((256, 256, 1)) 
model.compile(optimizer='adam', loss='mse')