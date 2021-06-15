import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

from tensorflow import keras
### hack tf-keras to appear as top level keras
import sys
sys.modules['keras'] = keras
### end of hack

from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics

from PIL import Image
import os
import yaml

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

##################################################################

config_filename = 'config.yaml'

images_folder = ''
images_shape = (16, 16)

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    data_folder = config['data_folder']
    images_folder = os.path.join(data_folder, config['images_folder'])
    images_shape = config['images_shape']

    width = images_shape['width'] if 'width' in images_shape else 16
    height = images_shape['height'] if 'height' in images_shape else 16
    images_shape = (width, height)

    saves_folder = config['saves_folder']

# defining the key parameters
colors_dim = 3 # rgb

# Parameters of the input images (handwritten digits)
original_dim = images_shape[0]*images_shape[1]*colors_dim

# Latent space is of dimension 2.  This means that we are reducing the dimension from 784 to 2
latent_dim = 2
intermediate_dim = 256
batch_size = 50
epochs = 10000
epsilon_std = 1.0

##################################################################

def sampling(args: tuple):
    # we grab the variables from the tuple
    z_mean, z_log_var = args
    print(z_mean)
    print(z_log_var)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon  # h(z)

##################################################################

# input to our encoder
x = Input(shape=(original_dim,), name="input")
# intermediate layer
h = Dense(intermediate_dim, activation='relu', name="encoding")(x)
# defining the mean of the latent space
z_mean = Dense(latent_dim, name="mean")(h)
# defining the log variance of the latent space
z_log_var = Dense(latent_dim, name="log-variance")(h)
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# defining the encoder as a keras model
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
# print out summary of what we just did
encoder.summary()

##################################################################

# Input to the decoder
input_decoder = Input(shape=(latent_dim,), name="decoder_input")
# taking the latent space to intermediate dimension
decoder_h = Dense(intermediate_dim, activation='relu', name="decoder_h")(input_decoder)
# getting the mean from the original dimension
x_decoded = Dense(original_dim, activation='sigmoid', name="flat_decoded")(decoder_h)
# defining the decoder as a keras model
decoder = Model(input_decoder, x_decoded, name="decoder")
decoder.summary()

##################################################################

# grab the output. Recall, that we need to grab the 3rd element our sampling z
output_combined = decoder(encoder(x)[2])
# link the input and the overall output
vae = Model(x, output_combined)
# print out what the overall model looks like
vae.summary()

##################################################################

def vae_loss(x: tf.Tensor, x_decoded_mean: tf.Tensor):
  # Aca se computa la cross entropy entre los "labels" x que son los valores 0/1 de los pixeles, y lo que salió al final del Decoder.
  xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) # x-^X
  kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
  vae_loss = K.mean(xent_loss + kl_loss)
  return vae_loss

vae.compile( loss=vae_loss,experimental_run_tf_function=False)
vae.summary()

##################################################################

if not os.path.exists(images_folder) or not os.path.isdir(images_folder):
    print(f'ERROR: Missing images folder')

images = [file for file in os.listdir(images_folder) if file.endswith(('jpeg', 'png', 'jpg'))]

x = []

processed = 0
for image in images:
    full_path = os.path.join(images_folder, image)

    img = Image.open(full_path)

    if len(img.split()) >= 3:
        # img.thumbnail(images_shape)
        img = img.convert("RGB")
        img = img.resize(images_shape)
        img = np.asarray(img, dtype=np.float32) / 255
        img = img[:, :, :3]

        x.append(img)

        processed += 1

print(f"Loaded {processed} out of {len(images)} images with shape {images_shape}")

x = np.array(x)

(x_train, y_train), (x_test, y_test) = (x, x), (x, x)

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

##################################################################

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

##################################################################    

# It is not really useful here because we do not have labels for this images, they are all different

# colors = [i for i in range(x_train.shape[0])]
# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=colors, cmap='viridis')
# plt.colorbar()
# plt.show()

################################################################## 

n = 15  # figure with 15x15 digits
digit_size = images_shape[0]
figure = np.zeros((images_shape[0] * n, images_shape[1] * n, colors_dim))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(images_shape[0], images_shape[1], colors_dim)
        figure[i * images_shape[0]: (i + 1) * images_shape[0],
               j * images_shape[1]: (j + 1) * images_shape[1]] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

################################################################## 