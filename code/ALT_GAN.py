import os
from PIL import Image
import numpy as np
from keras.layers import Input, Dense, Flatten, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose, Dropout, Activation, GaussianNoise, AveragePooling2D
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam

class BDI_GAN():
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

        # ----- Downsample a 256x256 image -----
   
        # Downsample to 128x128
        model.add(Conv2D(filters=256, kernel_size=4, input_shape=[self.img_rows, self.img_cols, self.channels], padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(AveragePooling2D())

        # We Downsample to 64x64
        model.add(Conv2D(filters=128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(AveragePooling2D())

        # We Downsample to 32x32
        model.add(Conv2D(filters=64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(AveragePooling2D())

        # ----- Upsample again -----

        # Upsample to 64x64
        model.add(Conv2DTranspose(filters = 32, kernel_size = 3, padding = 'same'))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # Upsample to 128x128
        model.add(Conv2DTranspose(filters = 16, kernel_size = 3, padding = 'same'))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # Upsample to 256x256
        model.add(Conv2DTranspose(filters = 3, kernel_size = 3, padding = 'same'))   # filters = 3, to get 256x256x3
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
        model.add(GaussianNoise(0.2, [self.img_rows, self.img_cols, self.channels]))

        # 256x256x3
        model.add(Conv2D(filters=8, kernel_size=4, padding="same"))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D())

        # Downsample to 128x128
        model.add(Conv2D(filters=16, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum = norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D())

        # Downsample to 64x64
        model.add(Conv2D(filters=32, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D())

        # Downsample to 32x32
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D())

        # Downsample to 16x16
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D())

        # Downsample to 8x8
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=norm_momentum))
        model.add(LeakyReLU(relu_alpha))
        model.add(Dropout(drop_rate))
        model.add(AveragePooling2D())

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

    def train_discriminator(self, half_batch_size, X_paintings, X_photos):
        # Select random real paintings
        idx_paintings = np.random.randint(0, X_paintings.shape[0], half_batch_size)
        real_paintings = X_paintings[idx_paintings]

        # Select random photos for generator input
        idx_photos = np.random.randint(0, X_photos.shape[0], half_batch_size)
        input_photos = X_photos[idx_photos]

        # Generate fake paintings
        fake_paintings = self.generator.predict(input_photos)

        # Combine real and fake paintings
        combined_imgs = np.concatenate([real_paintings, fake_paintings], axis=0)

        # Create corresponding labels
        real_labels = np.ones((half_batch_size, 1))
        fake_labels = np.zeros((half_batch_size, 1))
        combined_labels = np.concatenate([real_labels, fake_labels], axis=0)

        # Train the discriminator
        (loss, accuracy) = self.discriminator.train_on_batch(combined_imgs, combined_labels)
        return loss, accuracy
    
    def train_generator(self, half_batch_size, X_photos):
        idx_photos = np.random.randint(0, X_photos.shape[0], half_batch_size)
        input_photos = X_photos[idx_photos]

        valid_labels = np.ones((half_batch_size, 1))
        loss = self.combined.train_on_batch(input_photos, valid_labels)

        return loss

    def train_model(self, epochs, batch_size=128, output_image_interval=50, save_model_interval=1000, input_dir="data", output_dir="images", print_interval=0):
        # Load the dataset of photos and preprocess them
        X_photos = self.load_training_data(image_folder=f"{input_dir}/photo_jpg", image_size=(self.img_rows, self.img_cols))

        # Load the dataset of paintings as the discriminator's real inputs
        X_paintings = self.load_training_data(image_folder=f"{input_dir}/monet_jpg", image_size=(self.img_rows, self.img_cols))

        # Prepare temp folder for continuous output
        temp_dir = f"{output_dir}/temp"

        # We divide our batches 50/50 into real and fake images
        half_batch_size = int(batch_size / 2)

        for epoch in range(epochs):
            # ------------------ Train Discriminator ------------------- 
            d_loss, d_accuracy = self.train_discriminator(half_batch_size, X_paintings, X_photos)

            # ------------------ Train Generator ------------------- 
            g_loss = self.train_generator(half_batch_size, X_photos)

            self.discriminator_real_losses.append(d_loss)
            self.disciminator_fake_losses.append(d_accuracy)
            self.generator_losses.append(g_loss)

           # ------------------ Output Data from training ------------------- 

            if(print_interval > 0):
                # Plot the progress
                if(epoch % print_interval == 0):
                    print("------------------------------------------------------")
                    print(f"                Epoch {epoch}")
                    print(f"Discriminator real loss:  {d_loss:.5f}")
                    print(f"Discriminator fake loss:  {d_accuracy:.5f}")
                    print(f"Generator loss:           {g_loss:.5f}")
                    print("------------------------------------------------------")

                    # ------------------ Print Predictions for 5 real and fake images ------------------- 

                    # Get discriminator's predictions for a sample of generated images
                    sample_noise = np.random.normal(0, 1, (5, self.latent_dim))
                    sample_gen_imgs = self.generator.predict(sample_noise, verbose=0)
                    predictions = self.discriminator.predict(sample_gen_imgs, verbose=0)

                    # Print the first few predictions
                    print("5 generated images:")
                    print(predictions.flatten())  # Flatten to make it easier to read

                    # Also sample real images and get their predicions
                    idx = np.random.randint(0, X_paintings.shape[0], 5)
                    sample_real_imgs = X_paintings[idx]
                    predictions = self.discriminator.predict(sample_real_imgs, verbose=0)

                    # Print the first few predictions
                    print("5 real images:")
                    print(predictions.flatten())  # Flatten to make it easier to read
                    print("------------------------------------------------------")


                # If at save interval, save generated images
                if (epoch % output_image_interval == 0 and output_image_interval > 0):
                    self.save_generated_image(epoch, output_dir=f"{temp_dir}/images", name=f"img_gen_{epoch}")

                # If at save interval, save the model
                if (epoch % save_model_interval == 0 and save_model_interval > 0):
                    self.save_model(path=f"{temp_dir}/models", epoch=epoch)
