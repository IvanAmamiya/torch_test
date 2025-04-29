import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os

from app.models.cgan import Generator, Discriminator
from app.data_loader import CIFAR10Loader # Use the existing loader, but maybe without Mixup

# --- Configuration ---
latent_dim = 100
num_classes = 10
img_shape = (3, 32, 32)
batch_size = 64
lr = 0.0002
b1 = 0.5 # Adam beta1
b2 = 0.999 # Adam beta2
n_epochs = 200 # Adjust as needed
sample_interval = 1000 # Save generated images every N batches
checkpoint_interval = 10000 # Save model checkpoints every N batches
output_dir = "generated_images_cgan"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Models and Loss ---
generator = Generator(latent_dim, num_classes, img_shape).to(device)
discriminator = Discriminator(num_classes, img_shape).to(device)
adversarial_loss = nn.BCEWithLogitsLoss().to(device) # More stable than BCELoss + Sigmoid

# --- Optimizers ---
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# --- Data Loader ---
# Note: We might want a version of the loader *without* Mixup for standard GAN training
# For now, let's use the existing one but access the raw loader
cifar_loader = CIFAR10Loader(batch_size)
# Access the underlying DataLoader directly to avoid Mixup for GAN training
train_loader = cifar_loader.train_loader

# --- Training Loop ---
print(f"Starting cGAN training on {device}...")
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(train_loader):

        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1, device=device, dtype=torch.float32)
        fake = torch.zeros(imgs.size(0), 1, device=device, dtype=torch.float32)

        # Configure input
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (imgs.size(0),), device=device)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid) # Train G to make D output 'valid'

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        real_pred = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred = discriminator(gen_imgs.detach(), gen_labels) # Detach G's output
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Log Progress
        # ---------------------
        batches_done = epoch * len(train_loader) + i
        if batches_done % 100 == 0: # Print less frequently
            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

        if batches_done % sample_interval == 0:
            print(f"Saving generated image sample at batch {batches_done}...")
            # Generate images for a fixed set of noise and labels for comparison
            fixed_noise = torch.randn(num_classes * 10, latent_dim, device=device) # 10 samples per class
            fixed_labels = torch.repeat_interleave(torch.arange(num_classes, device=device), 10)
            with torch.no_grad():
                gen_imgs_sample = generator(fixed_noise, fixed_labels).detach().cpu()
            # Rescale images from [-1, 1] to [0, 1] for saving
            gen_imgs_sample = (gen_imgs_sample + 1) / 2
            vutils.save_image(gen_imgs_sample, f"{output_dir}/epoch_{epoch}_batch_{batches_done}.png", nrow=num_classes, normalize=False)

        if batches_done % checkpoint_interval == 0 and batches_done > 0:
            print(f"Saving model checkpoint at batch {batches_done}...")
            torch.save(generator.state_dict(), f"generator_epoch_{epoch}_batch_{batches_done}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}_batch_{batches_done}.pth")

print("cGAN Training Finished.")
# Save final models
torch.save(generator.state_dict(), "cgan_generator_final.pth")
torch.save(discriminator.state_dict(), "cgan_discriminator_final.pth")
print("Final cGAN models saved.")
