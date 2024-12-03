from AEGAN import AEGAN

# Running the code
generator_lr = 1e-5
discriminator_lr = 1e-6

# We note our model
version = "AEGAN"

for generator_lr in [1e-2, 5e-5, 1e-6]:
    for discriminator_lr in [1e-2, 5e-5, 1e-6]:
        # Create and build model
        model = AEGAN(g_lr=generator_lr, d_lr=discriminator_lr)
        build_gan = model.build_model(show_summary=False)

        # Train model
        model.train_model(
            epochs=1000, 
            batch_size=32, 
            output_image_interval = 100, 
            save_model_interval = 1000, 
            input_dir="data/monet_jpg", 
            output_dir=f"output/{version}__g_lr_{generator_lr:1e}___d_lr_{discriminator_lr:1e}",
            print_interval=10)