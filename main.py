import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
from datetime import datetime


start_time = datetime.now()   # get now time

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256 * 16 * 16),  # Start from a random noise vector
            nn.ReLU(),
            nn.Unflatten(1, (256, 16, 16)),  # Reshape to (256, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (N, 256, 16, 16) to (N, 128, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # (N, 128, 32, 32) to (N, 64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),      # (N, 64, 64, 64) to (N, 3, 128, 128)
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),      # (N, 3, 128, 128) to (N, 64, 64, 64)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),     # (N, 64, 64, 64) to (N, 128, 32, 32)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),    # (N, 128, 32, 32) to (N, 256, 16, 16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),    # (N, 256, 16, 16) to (N, 512, 8, 8)
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),      # Adjust according to the final output size
            nn.Sigmoid()                     # Output values between 0 and 1
        )

    def forward(self, img):
        return self.model(img)

# Create output directory
output_dir = './images'
checkpoint_dir = './checkpoints'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Load and preprocess your image (ensure this file exists)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load your sample image
img = Image.open('./image.png').convert('RGB')  # Ensure the image is read in RGB mode
img = transform(img).unsqueeze(0)  # Add batch dimension (now shape: [1, 3, 128, 128])

# Create a batch of images from the single image
batch_size = 64
real_images = img.repeat(batch_size, 1, 1, 1)  # Shape is [64, 3, 128, 128]

# Hyperparameters
num_epochs = 100
num_samples = 20
latent_dim = 100
resume_training = False

# Create models
generator = Generator()
discriminator = Discriminator()

# Optimizers!!!
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Load checkpoint if resuming i think this part have a bug sorry <3
if resume_training:
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}...")
else:
    start_epoch = 0

for epoch in range(start_epoch, num_epochs):
    for i in range(num_samples):
        # random noise
        os.system("cls")
        current_time = datetime.now()
        elapsed_time = current_time - start_time
        days, seconds = elapsed_time.days, elapsed_time.seconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"generating "+("■"*i)+f" {i*5+5}%")
        print(f"Uptime: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds", end='\r')
        z = torch.randn(batch_size, latent_dim)

        gen_imgs = generator(z)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_images), real_labels)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(gen_imgs), real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Save generated images periodically
        if (epoch * num_samples + i) % 10 == 0:  # Save every 10 samples
            img_name = os.path.join(output_dir, f"image_{epoch * num_samples + i}.png")
            save_image(gen_imgs.data[:16], img_name, nrow=4, normalize=True)

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, os.path.join(checkpoint_dir, 'checkpoint.pth'))
    print(f"Checkpoint saved at epoch {epoch + 1}.")

print("Training complete. Images saved to './images/'.")
