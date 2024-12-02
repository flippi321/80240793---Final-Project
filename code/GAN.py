from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, Dropout, Activation, ReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

class DCGAN():
    def __init__(self):
        # Image size settings
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  # Our images are 256x256 images
        
        # General GAN settings
        self.latent_dim = 100
        self.label_smoothing = 0.2
        self.label_smoothing_end = 1000

        # Generator settings
        self.batch_normalisation_momentum = 0.9
        self.g_lr = 2e-5

        # Discriminator settings
        self.relu_alpha=0.2
        self.d_lr = 2e-5

        self.discriminator_losses = []
        self.generator_losses = []
    
    def build_gan(self):
        # Initialize the optimizers
        discriminator_optimizer = Adam(learning_rate=self.d_lr, beta_1=0.5, beta_2=0.99)
        combined_optimizer = Adam(learning_rate=self.g_lr, beta_1=0.5, beta_2=0.99)

        # Build the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=combined_optimizer)

    def build_generator(self):
        model = Sequential()
        
        # Starting from a 8x8 dimension, upscale input from latent space
        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        
        # Upsample to 16x16
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Upsample to 32x32
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Upsample to 64x64
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Upsample to 64x64
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Upsample to 128x128
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Upsample to 256x256
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Final Conv2D layer to set the number of channels
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        # We start with a photo of size 256x256
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=self.relu_alpha))
        model.add(Dropout(0.4))

        # Downsample to 128x128
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=self.relu_alpha))
        model.add(Dropout(0.4))

        # Downsample to 64x64
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=self.relu_alpha))
        model.add(Dropout(0.25))

        # Downsample to 32x32
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=self.relu_alpha))
        model.add(Dropout(0.25))

        # Downsample to 16x16
        model.add(Conv2D(1024, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=self.relu_alpha))
        model.add(Dropout(0.4))

        # Flatten and output layer for binary classification
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def save_model(self, path):
        self.discriminator.save(path + "/discriminator.keras")
        self.generator.save(path + "/generator.keras")
        self.combined.save(path + "/combined.keras")

    def load_gan(self, path):
        self.discriminator = load_model(path + "/discriminator.keras")
        self.generator = load_model(path + "/generator.keras")
        self.combined = load_model(path + "/combined.keras")

    