from GAN import BDI_GAN

# Running the code
model = BDI_GAN()

print("Building Model...")
build_gan = model.build_model()
print("Done :-) \n")

# We train the model for 10000 epochs
print("Training Model...")
model.train_model(epochs=10000, batch_size=32, save_interval=50, input_dir="data/monet_jpg", print_interval=10)