# Import necessary modules
from pathlib import Path
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from medical_diffusion.data.datasets.dataset_simple_2d import SimpleDataset2D
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.embedders.latent_embedders import VQVAE

import pandas as pd
from torchvision import transforms as T
from PIL import Image
import os

# One-hot mapping for locations
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

class UltrasoundDataset(SimpleDataset2D):
    def __init__(self, csv_file, path_root, augment_horizontal_flip=False, augment_vertical_flip=False):
        super().__init__(path_root)
        self.labels = pd.read_csv(csv_file)
        self.item_pointers = self.labels['Image Name'].tolist()
        self.descriptions = self.labels['Location'].tolist()

        # Define transformations, including optional augmentations
        transform_list = [T.Resize((128, 128))]
        if augment_horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip())
        if augment_vertical_flip:
            transform_list.append(T.RandomVerticalFlip())
        transform_list.extend([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])
        self.transform = T.Compose(transform_list)

    def __getitem__(self, index):
        image_name = self.item_pointers[index]
        description = self.descriptions[index]

        # Load the image
        path_item = Path(self.path_root) / f"{image_name}"
        img = self.load_item(path_item)

        # Apply transformations
        img = self.transform(img)

        # One-hot encode the description
        one_hot_condition = torch.tensor(location_to_one_hot[description], dtype=torch.float32)

        # Return the image and its condition
        return {'uid': image_name, 'source': img, 'condition': one_hot_condition}

    def load_item(self, path_item):
        try:
            return Image.open(path_item).convert('RGB')
        except Exception as e:
            print(f"Error loading image {path_item}: {e}")
            raise

# Modify VQVAE to include condition handling
class ModifiedVQVAE(VQVAE):
    def decode(self, z, condition=None):
        z, _ = self.quantizer(z)

        if condition is not None:
            # Expand condition to match spatial dimensions of z
            condition = condition.view(condition.size(0), -1, 1, 1).expand(-1, -1, z.size(2), z.size(3))
            # Concatenate z and condition along the channel dimension
            z = torch.cat([z, condition], dim=1)

        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i - 1](h)
        x = self.outc(h)
        return x

    def forward(self, x_in, condition=None):
        # Encoder
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # Quantizer
        z_q, emb_loss = self.quantizer(z)

        # Decoder
        if condition is not None:
            # Expand condition to match spatial dimensions of z_q
            condition = condition.view(condition.size(0), -1, 1, 1).expand(-1, -1, z_q.size(2), z_q.size(3))
            # Concatenate z_q and condition along the channel dimension
            z_q = torch.cat([z_q, condition], dim=1)

        h = self.inc_dec(z_q)
        for i in range(len(self.decoders) - 1, -1, -1):
            h = self.decoders[i](h)
        out = self.outc(h)

        return out, emb_loss

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        x = batch['source']
        condition = batch['condition']
        target = x

        pred, emb_loss = self.forward(x, condition)

        # Compute Loss
        loss = self.loss_fct(pred, target) + emb_loss * self.embedding_loss_weight

        # Log Scalars
        self.log(f"{state}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

# Main training script
if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None

    print("Loading Data")
    ds = UltrasoundDataset(
        csv_file='/home/maryam.arjemandi/Documents/medfusion/train.csv',
        path_root='/l/users/maryam.arjemandi/New_train_flat/',
        augment_horizontal_flip=True,
        augment_vertical_flip=True
    )
    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=4,
        pin_memory=True
    )

    print("Initializing Model")
    model = ModifiedVQVAE(
        in_channels=3,
        out_channels=3,
        emb_channels=8,
        spatial_dims=2,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        num_embeddings=8192,
        beta=0.25,
        embedding_loss_weight=1e-6,
        use_attention='none',
        loss=torch.nn.MSELoss
    )

    print("Setting up Trainer")
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor="train/loss",  # Replace with the metric you want to monitor
        save_last=True,
        save_top_k=5,
        mode="min",  # Set to "min" for metrics where lower is better (e.g., loss)
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=gpus,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        max_epochs=100,
        limit_val_batches=0
    )

    print("Starting Training")
    trainer.fit(model, datamodule=dm)

    # Save Model
    torch.save(model.state_dict(), str(path_run_dir / 'best_latent_embedder_model.pth'))
    print("Training Complete.")