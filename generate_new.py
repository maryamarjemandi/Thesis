import torch
from torchvision.utils import save_image
from pathlib import Path
from medical_diffusion.models.embedders.latent_embedders import VAE

# Load the stored latent `z_value`
z_value = torch.load('/home/maryam.arjemandi/Documents/medfusion/z_value.pt')

# Check the original shape of `z_value`
print("Original shape of z_value:", z_value.shape)

# Adjust the shape based on the actual dimensions of `z_value`
z_value = z_value.view(1, 8, 16, 16)  # Assuming 8 channels, adjust if necessary

if __name__ == "__main__":
    output_dir = Path.cwd() / 'sanity_check_image'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = VAE.load_from_checkpoint('/home/maryam.arjemandi/Documents/medfusion/runs/2024_10_16_223153/last.ckpt')
    model.eval()

    with torch.no_grad():
        # Decode using the specified `z_value`
        generated_image = model.decode(z_value)

        # Min-Max normalization to ensure pixel values are in [0, 1]
        generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())

        # Save the generated image
        save_image(generated_image, output_dir / 'generated_sanity_check_image.png')

    print("Image generated using the stored z value.")
