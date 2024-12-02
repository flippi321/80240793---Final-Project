from GAN import BDI_GAN

# Running the code
model = BDI_GAN()

build_gan = model.build_model(show_summary=True)

# Train model
model.train_model(
    epochs=10000, 
    batch_size=64, 
    output_image_interval = 50, 
    save_model_interval = 1000, 
    input_dir="data/monet_jpg", 
    output_dir="output", 
    print_interval=10)