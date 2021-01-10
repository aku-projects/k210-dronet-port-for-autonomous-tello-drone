import tensorflow as tf
import keras
import cv2
from PIL import Image 
import numpy as np
from keras.models import Model, model_from_json, save_model
from keras.layers import Reshape, Input, Lambda, Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers

weights_path= "./dronet/best_weights.h5"
json_model_path="./dronet/model_struct.json"

with open(json_model_path, 'r') as json_file:
    loaded_model_json = json_file.read()
oldModel = model_from_json(loaded_model_json)
oldModel.load_weights(weights_path)
oldModel.compile(loss='mse', optimizer='sgd')
oldModel.summary()

output_dim = 1
img_input = Input(shape=(200, 200, 1))
x1 = Lambda(lambda x: x*(1.0/255.0))(img_input)
x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(x1)

x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

# First residual block
x2 = keras.layers.normalization.BatchNormalization()(x1)

x2 = Activation('relu')(x2)

x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4))(x2)

x2 = keras.layers.normalization.BatchNormalization()(x2)

x2 = Activation('relu')(x2)


x2 = Conv2D(32, (3, 3), padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4))(x2)

x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)

x3 = add([x1, x2])


# Second residual block
x4 = keras.layers.normalization.BatchNormalization()(x3)
x4 = Activation('relu')(x4)
x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4))(x4)

x4 = keras.layers.normalization.BatchNormalization()(x4)
x4 = Activation('relu')(x4)
x4 = Conv2D(64, (3, 3), padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4))(x4)

x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
x5 = add([x3, x4])

# Third residual block
x6 = keras.layers.normalization.BatchNormalization()(x5)
x6 = Activation('relu')(x6)
x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4))(x6)

x6 = keras.layers.normalization.BatchNormalization()(x6)
x6 = Activation('relu')(x6)
x6 = Conv2D(128, (3, 3), padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4))(x6)

x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
x7 = add([x5, x6])

x = Flatten()(x7)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

# Steering channel
steer = Dense(output_dim)(x)

# Collision channel
coll = Dense(output_dim)(x)
coll = Activation('sigmoid')(coll)

# Define steering-collision model
newModel = Model(inputs=[img_input], outputs=[steer, coll])
newModel.summary()

newModel.set_weights(oldModel.get_weights())


model_json = newModel.to_json()
with open("./dronet/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
newModel.save_weights("./dronet/model_weights.h5")
print("Saved model to disk")
save_model(newModel,'./dronet/model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(newModel)
tfmodel = converter.convert()
open ("./dronet/model.tflite" , "wb") .write(tfmodel)



newModel.compile(loss='mse', optimizer='sgd')
img = Image.open('./images/1.jpg')
img = np.asarray(img)
cv_image = np.asarray(img.reshape(200,200,1))
outs = newModel.predict_on_batch([cv_image[None]])
steer, coll = outs[0][0], outs[1][0]
print(steer)
print(coll)
