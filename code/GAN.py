import os
from PIL import Image
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, Dropout, Activation, ReLU
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam

class BDI_GAN():
    def __init__(self):
        # Image size settings
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  # Our images are 256x256 images
        
        # General GAN settings
        self.latent_dim = 4096
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
    
    def build_model(self, show_summary=False):
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
        
        # Starting from an 8x8 dimension, upscale input from latent space
        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        
        # Upsample to 16x16
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Upsample to 32x32
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
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=self.batch_normalisation_momentum))
        model.add(ReLU())

        # Upsample to 256x256
        model.add(UpSampling2D())
        model.add(Conv2D(8, kernel_size=3, padding="same"))
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

    def load_training_data(self, image_folder="data", image_size=(256, 256)):
        images = []
        # We add all images to the array
        for img_file in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_file)
            try:
                img = load_img(img_path, target_size=image_size, color_mode='rgb')
                img = img_to_array(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        return np.array(images)
    
    def generate_images(self, n_images=1):
        noise = np.random.normal(0, 1, (n_images, self.latent_dim))
        return self.generator.predict(noise)

    def save_generated_image(self, epoch, output_dir="images", name="img_gen_temp"):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a single image
        gen_img = self.generate_images(n_images=1)[0]  # Take the first generated image
        
        # Rescale image values
        gen_img = 0.5 * gen_img + 0.5               # Rescale from [-1, 1] to [0, 1]
        gen_img = (gen_img * 255).astype(np.uint8)  # Rescale from [0, 1] to  [0, 255]
        
        # Save the image in the output directory
        pil_img = Image.fromarray(gen_img, mode='RGB')
        save_path = os.path.join(output_dir, f"{name}_epoch_{epoch}.png")
        pil_img.save(save_path)
        print(f"Saved generated image at: {save_path}")


    def train_model(self, epochs, batch_size=128, save_interval=50, input_dir="data", output_dir="images", print_interval=0):
        # Load the dataset
        X_train = self.load_training_data(image_folder=input_dir, image_size=(self.img_rows, self.img_cols))

        # Prepare temp folder for continious output
        temp_dir = f"{output_dir}/temp"

        # We divide our batches 50/50 into real and fake images
        half_batch_size = int(batch_size / 2)

        for epoch in range(epochs):
            # ------------------- Train Discriminator ------------------- 

            # Select a random batch of real images
            # TODO Replace with sample?
            idx = np.random.randint(0, X_train.shape[0], half_batch_size)
            imgs = X_train[idx]

            # Create a batch of fake images
            gen_imgs = self.generate_images(half_batch_size)

            # We divide the batch size into real and fake images
            real_labels = np.ones((half_batch_size, 1))
            fake_labels = np.zeros((half_batch_size, 1))           

            # TODO Might be smart to add label smoothing, check this

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------ Train Generator ------------------- 

            # Generate noise for the generator
            noise = np.random.normal(0, 1, (half_batch_size, self.latent_dim))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, real_labels)

            self.discriminator_losses.append(d_loss[0])
            self.generator_losses.append(g_loss)

            # ------------------ Output Data from training ------------------- 

            if(print_interval > 0):
                # Plot the progress
                if(epoch % print_interval == 0):
                    accuracy = d_loss[1]

                    print(f"                  Epoch {epoch}")
                    print(f"Discriminator loss:       {d_loss[0]:.5f}")
                    print(f"Discriminator accuracy:   {accuracy:%}")
                    print(f"Generator loss:           {g_loss:.5f}")
                    print("------------------------------------------------------")

                    # Get discriminator's predictions for a sample of generated images
                    sample_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    sample_gen_imgs = self.generator.predict(sample_noise, verbose=0)
                    predictions = self.discriminator.predict(sample_gen_imgs, verbose=0)

                    # Print the first few predictions
                    print("Sample of discriminator's predictions on generated images (first 5):")
                    print(predictions[:5].flatten())  # Flatten to make it easier to read
                    print("------------------------------------------------------")

                # TODO If at save interval, save generated images
                if (epoch % save_interval == 0):
                    self.save_generated_image(epoch, output_dir=temp_dir, name=f"img_gen_{epoch}")