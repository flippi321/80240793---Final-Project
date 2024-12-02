from GAN import BDI_GAN

# Running the code
generator_lr = 5e-2
discriminator_lr = 5e-2
model = BDI_GAN(g_lr=generator_lr, d_lr=discriminator_lr)

build_gan = model.build_model(show_summary=False)

# Train model
model.train_model(
    epochs=10000, 
    batch_size=32, 
    output_image_interval = 50, 
    save_model_interval = 1000, 
    input_dir="data/monet_jpg", 
    output_dir="output", 
    print_interval=10)