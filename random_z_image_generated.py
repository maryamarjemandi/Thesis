import torch
from torchvision.utils import save_image
from pathlib import Path
from medical_diffusion.models.embedders.latent_embedders import VAE

if __name__ == "__main__":
    output_dir = Path.cwd() / 'random_generated_image'
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VAE.load_from_checkpoint('/home/maryam.arjemandi/Documents/medfusion/runs/2024_10_16_223153/last.ckpt')
    model.eval()

    random_z = torch.randn(1, 8, 16, 16)

    with torch.no_grad():
        generated_image = model.decode(random_z)
        generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
        save_image(generated_image, output_dir / 'random_generated_image.png')

    print("Random image generated using a new random z value.")
