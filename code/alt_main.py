from ALT_GAN import ALT_GAN

# Running the code
generator_lr = 1e-5
discriminator_lr = 1e-6

# Create and build model
model = ALT_GAN(g_lr=generator_lr, d_lr=discriminator_lr)
build_gan = model.build_model(show_summary=False)

# Train model
model.train_model(
    epochs=1000, 
    batch_size=32, 
    output_image_interval = 100, 
    save_model_interval = 1000, 
    input_dir="data/", 
    output_dir=f"output/alt___g_lr_{generator_lr:1e}___d_lr_{discriminator_lr:1e}",
    print_interval=10)