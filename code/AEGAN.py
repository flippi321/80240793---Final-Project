import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.layers import Input, BatchNormalization, GaussianNoise, Dense, Reshape, Flatten, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose, Dropout, Activation, AveragePooling2D, MaxPooling2D
from keras._tf_keras.keras.models import Sequential, Model, load_model
from keras._tf_keras.keras.utils import img_to_array, load_img
from keras._tf_keras.keras.optimizers import Adam

class AEGAN():
    def __init__(self, g_lr=5e-5, d_lr=5e-5):
        # Image size settings
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  # Our images are 256x256 rgb images
        
        # General GAN settings
        self.latent_dim = 100
        self.label_smoothing = 0.2
        self.label_smoothing_end = 1000

        # Generator settings
        self.batch_normalisation_momentum = 0.9
        self.g_lr = g_lr

        # Discriminator settings
        self.discriminator_leaky_relu_alpha = 0.2
        self.discriminator_dropout_rate = 0.25
        self.d_lr = d_lr

        self.discriminator_real_losses = []
        self.discriminator_fake_losses = []
        self.generator_losses = []
    
    def build_model(self, show_summary=False):
        # Initialize the optimizers
        discriminator_optimizer = Adam(learning_rate=self.d_lr, beta_1=0.5, beta_2=0.99)
        combined_optimizer = Adam(learning_rate=self.g_lr, beta_1=0.5, beta_2=0.99)

        # Build the generator
        self.generator = self.build_generator(show_summary=show_summary)

        # Build the discriminator
        self.discriminator = self.build_discriminator(show_summary=show_summary)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

        # Input photo for generator
        photo = Input(shape=self.img_shape)
        painting = self.generator(photo)

        # For the combined model, only train the generator
        self.discriminator.trainable = False

        # The discriminator evaluates the painting
        validity = self.discriminator(painting)

        # Combined model: generator and discriminator
        self.combined = Model(photo, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=combined_optimizer)


    def build_generator(self, show_summary=False):
        model = Sequential()

        # Settings for the discriminator
        norm_momentum = 0.7

        # ----- Start with 256x256 image  -----
   
        # Downsample to 128x128
        model.add(Conv2D(filters=256, kernel_size=3, input_shape=[self.img_rows, self.img_cols, self.channels], padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # We Downsample to 64x64
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # We Downsample to 32x32
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # ----- Upsample again -----

        # Upsample to 64x64
        model.add(Conv2DTranspose(filters = 32, kernel_size = 3, padding = 'same'))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # Upsample to 128x128
        model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same'))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # Upsample to 256x256
        model.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same'))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # Finish with 256x256x3
        model.add(Conv2D(filters = self.channels, kernel_size = 3, padding = 'same'))   # filters = 3, to get 256x256x3
        model.add(Activation("sigmoid"))

        if(show_summary):
            model.summary()

        input_photo = Input(shape=self.img_shape)
        painting = model(input_photo)

        return Model(input_photo, painting)

    def build_discriminator(self, show_summary=False):
        model = Sequential()

        # Settings for the discriminator
        norm_momentum = 0.7
        relu_alpha = 0.2
        drop_rate = 0.25

        # Add Gaussian noise
        model.add(GaussianNoise(0.2, input_shape=[self.img_rows, self.img_cols, self.channels]))

        # 256x256x3
        model.add(Conv2D(filters=8, kernel_size=4, padding="same"))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        # Downsample to 128x128
        model.add(Conv2D(filters=16, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        # Downsample to 64x64
        model.add(Conv2D(filters=32, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        # Downsample to 32x32
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        # Downsample to 16x16
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        # Downsample to 8x8
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        # Downsample to 4x4
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(0.2))

        # Single-digit Classification
        model.add(Dense(1, activation='sigmoid'))

        if(show_summary):
            model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def save_model(self, path, epoch=0):
        output_dir = f"{path}/epoch_{epoch}/"
        os.makedirs(output_dir, exist_ok=True)
        self.discriminator.save(output_dir + "discriminator.keras")
        self.generator.save(output_dir + "generator.keras")
        self.combined.save(output_dir + "combined.keras")

    def load_gan(self, path):
        self.discriminator = load_model(path + "/discriminator.keras")
        self.generator = load_model(path + "/generator.keras")
        self.combined = load_model(path + "/combined.keras")

    def load_training_data(self, training_dir="data/monet_jpg", max_size=32, image_size=(256, 256)):
        # List all image files in the directory
        all_images = tf.io.gfile.listdir(training_dir)
        
        # Randomly select a batch of image files
        selected_files = tf.random.shuffle(all_images)[:max_size]

        def load_image(img_file):
            img_path = tf.strings.join([training_dir, img_file])
            try:
                img = tf.io.read_file(img_path)             # Read the image file
                img = tf.image.decode_jpeg(img, channels=3) # Decode the image
                img = tf.image.resize(img, image_size)      # Resize to the target image size
                img = img / 255.0                           # Normalize the image to [0, 1] range
                return img
            except tf.errors.NotFoundError:
                print(f"File not found: {img_path.numpy()}")
                return None

        # Load images in parallel
        images = tf.map_fn(load_image, selected_files, dtype=tf.float32)

        # Return the batch as a tensor
        return images
    
    def generate_images(self, input_photo, n_images=1):
        return self.generator.predict(input_photo)

    def save_generated_image(self, epoch, X_photos, output_dir="images", name="img_gen_temp"):
        # Select random photos for generator input
        idx_photos = tf.random.uniform([1], 0, tf.shape(X_photos)[0], dtype=tf.int32)
        input_photo = tf.gather(X_photos, idx_photos)

        # Ensure the output directory exists
        tf.io.gfile.makedirs(output_dir)

        # Generate a single image
        gen_img = self.generate_images(input_photo)[0]  # Take the first generated image

        # Rescale image values
        gen_img = 0.5 * gen_img + 0.5  # Rescale from [-1, 1] to [0, 1]
        gen_img = tf.clip_by_value(gen_img, 0.0, 1.0)  # Ensure values are in [0, 1] range

        # Convert the tensor to a uint8 format for saving
        gen_img = tf.image.convert_image_dtype(gen_img, dtype=tf.uint8)

        # Save the image using PIL
        pil_img = Image.fromarray(gen_img.numpy(), mode='RGB')
        save_path = os.path.join(output_dir, f"{name}_epoch_{epoch}.png")
        pil_img.save(save_path)
        print(f"Saved generated image at: {save_path}")

    def train_discriminator(self, half_batch_size, X_paintings, X_photos):
        # Select random real paintings and photos
        idx_paintings = tf.random.uniform([half_batch_size], 0, tf.shape(X_paintings)[0], dtype=tf.int32)
        real_paintings = tf.gather(X_paintings, idx_paintings)

        idx_photos = tf.random.uniform([half_batch_size], 0, tf.shape(X_photos)[0], dtype=tf.int32)
        input_photos = tf.gather(X_photos, idx_photos)

        # Generate fake paintings
        fake_paintings = self.generator(input_photos, training=False)

        # Combine real and fake data
        combined_imgs = tf.concat([real_paintings, fake_paintings], axis=0)

        # Labels for real (1) and fake (0) samples
        real_labels = tf.ones((half_batch_size, 1), dtype=tf.float32)
        fake_labels = tf.zeros((half_batch_size, 1), dtype=tf.float32)
        combined_labels = tf.concat([real_labels, fake_labels], axis=0)

        # Train discriminator
        loss, accuracy = self.discriminator.train_on_batch(combined_imgs, combined_labels)
        return loss, accuracy

    def train_generator(self, half_batch_size, X_photos):
        # Select random photos
        idx_photos = tf.random.uniform([half_batch_size], 0, tf.shape(X_photos)[0], dtype=tf.int32)
        input_photos = tf.gather(X_photos, idx_photos)

        # Labels for valid (1) samples
        valid_labels = tf.ones((half_batch_size, 1), dtype=tf.float32)

        # Train generator via combined model
        loss = self.combined.train_on_batch(input_photos, valid_labels)
        return loss

    def train_model(self, epochs, batch_size=128, output_image_interval=50, save_model_interval=1000, input_dir="data", output_dir="images", print_interval=0):
        half_batch_size = batch_size // 2

        print("Loading data...")
        X_photos = self.load_training_data(training_dir=f"{input_dir}/photo_jpg_reduced", max_size=300, image_size=(self.img_rows, self.img_cols))
        X_paintings = self.load_training_data(training_dir=f"{input_dir}/monet_jpg", max_size=300, image_size=(self.img_rows, self.img_cols))
        print("Done!\nTraining model...")

        for epoch in range(epochs):
            # Train discriminator
            d_loss, d_accuracy = self.train_discriminator(half_batch_size, X_paintings, X_photos)
            # Train generator
            g_loss = self.train_generator(half_batch_size, X_photos)

            # Record losses
            self.discriminator_real_losses.append(d_loss)
            self.discriminator_fake_losses.append(d_accuracy)
            self.generator_losses.append(g_loss)

            # Print progress
            if print_interval > 0 and epoch % print_interval == 0:
                print(f"Epoch {epoch} - D Loss: {d_loss:.5f}, D Accuracy: {d_accuracy:.5f}, G Loss: {g_loss[0]:.5f}")

            # Save generated images periodically
            if epoch % output_image_interval == 0:
                self.save_generated_image(epoch, X_photos, output_dir=f"{output_dir}/images", name=f"img_gen_{epoch}")

            # Save model periodically
            if epoch % save_model_interval == 0:
                self.save_model(path=f"{output_dir}/models", epoch=epoch)