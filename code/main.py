from AEGAN import AEGAN
from GAN import BDI_GAN

# Running the code
generator_lr = 1e-5
discriminator_lr = 1e-6

# We note our model
version = "AEGAN"

# Create and build model
if version == "AEGAN":
    model = AEGAN(g_lr=generator_lr, d_lr=discriminator_lr)
else:
    model = BDI_GAN(g_lr=generator_lr, d_lr=discriminator_lr)


build_gan = model.build_model(show_summary=True)

# Train model
model.train_model(
    epochs=1000, 
    batch_size=16, 
    output_image_interval = 100, 
    save_model_interval = 1000, 
    input_dir="data", 
    output_dir=f"output/{version}__g-lr_{generator_lr:1e}___d-lr_{discriminator_lr:1e}",
    print_interval=10)    