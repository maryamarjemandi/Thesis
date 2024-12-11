import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms as T
from pytorch_lightning import LightningModule
from PIL import Image

# Dictionary for one-hot encoding based on prompt
location_to_one_hot = {
    "spine": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "femur": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "profile": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Heart(4ch)": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "lips&nose": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "kidney": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Heart(orbit)": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "heart(lvot)": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "heart(rvot)": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "heart(3vv)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "Diaphragm": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "abdomen": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "feet": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "cord": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "brain": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

class ConditionalVAE(nn.Module):
    def __init__(self, in_channels=3, emb_channels=8, condition_dim=15, **kwargs):
        super(ConditionalVAE, self).__init__()
        self.condition_embedding = nn.Linear(condition_dim, emb_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, emb_channels)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_channels * 2, 128 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Ensuring 3 output channels for RGB
            nn.Tanh()
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, x, condition):
        condition_embed = self.condition_embedding(condition)
        # x_encoded = self.encoder(x)
        print('x.shape', x.shape)
        print('condition_embed.shape', condition_embed.shape)
        combined_features = torch.cat([x, condition_embed], dim=-1)
        reconstruction = self.decoder(combined_features)
        return reconstruction

class ConditionalVAE_Lit(LightningModule):
    def __init__(self, in_channels=3, emb_channels=8, condition_dim=15):
        super().__init__()
        self.model = ConditionalVAE(in_channels, emb_channels, condition_dim)
    
    def forward(self, x, condition):
        return self.model(x, condition)
    
    def compute_loss(self, x, reconstruction):
        return self.model.mse_loss(reconstruction, x)

# Load the model checkpoint
checkpoint_path = '/home/maryam.arjemandi/Documents/medfusion/runs/2024_11_14_131955/last.ckpt'
model = ConditionalVAE_Lit(in_channels=3, emb_channels=8, condition_dim=15)
print("Loading model from checkpoint...")
model = model.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
print("Model loaded successfully.")
model.eval()

def generate_image(prompt, save_path):
    print(f"Generating image for prompt: {prompt}")
    condition_vector = torch.tensor(location_to_one_hot.get(prompt, [0]*15), dtype=torch.float32).unsqueeze(0)
    latent_vector = torch.rand((1, 8, 16, 16))
    print
    if torch.cuda.is_available():
        print("Using CUDA")
        model.cuda()
        latent_vector = latent_vector.cuda()
        condition_vector = condition_vector.cuda()

    with torch.no_grad():
        generated_image = model(latent_vector, condition_vector)
        print("Image generated with model")

    # Check the shape of generated_image
    print("Generated image shape:", generated_image.shape)
    
    generated_image = generated_image.cpu().squeeze(0).permute(1, 2, 0)
    generated_image = (generated_image + 1) / 2  # Normalize to [0,1] for saving as image
    generated_image = T.ToPILImage()(generated_image)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    generated_image.save(save_path)
    print(f"Image saved at {save_path}")

# Run the function to generate and save the image
prompt = "spine"
save_path = f"/home/maryam.arjemandi/Documents/medfusion/generated_{prompt}.png"
generate_image(prompt, save_path)