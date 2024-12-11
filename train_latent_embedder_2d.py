# Import necessary modules
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from medical_diffusion.data.datasets.dataset_simple_2d import SimpleDataset2D, UltrasoundDataset
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.embedders.latent_embedders import VAE, VQVAE, VAEGAN

import pandas as pd
from torchvision import transforms as T
from PIL import Image

# Define the custom dataset class
class UltrasoundDataset(SimpleDataset2D):
    def __init__(self, csv_file, path_root, augment_horizontal_flip=False, augment_vertical_flip=False, *args, **kwargs):
        super().__init__(path_root, *args, **kwargs)
        self.labels = pd.read_csv(csv_file)
        
        # Use the 'Image_name' column for image paths and 'Description' for labels
        self.item_pointers = self.labels['Image_name'].tolist()
        self.descriptions = self.labels['Description'].tolist()

        # Define transformations, including optional augmentations
        transform_list = []
        transform_list.append(T.Resize((128, 128)))  # Resize as an example, adjust as needed
        
        if augment_horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip())
        
        if augment_vertical_flip:
            transform_list.append(T.RandomVerticalFlip())
        
        transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)  # Normalization to [-1, 1]
        ])
        
        self.transform = T.Compose(transform_list)

    def __getitem__(self, index):
        image_name = self.item_pointers[index]
        description = self.descriptions[index]

        # Load the image
        path_item = Path(self.path_root) / f"{image_name}.png"  # Assuming images are in .png format
        img = self.load_item(path_item)

        # Apply transformations
        img = self.transform(img)

        # Return the image and its description (if needed, otherwise just the image)
        return {'uid': image_name, 'source': img, 'description': description}
    
    def load_item(self, path_item):
        # Loading image using PIL and converting to RGB
        return Image.open(path_item).convert('RGB')


# Main training script
if __name__ == "__main__":
    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None
    print("Loading Data")
    
    # ------------ Load Data ----------------
    ds = UltrasoundDataset(
        csv_file='/home/maryam.arjemandi/Desktop/Updated_Paraphrased_Ultrasound_Descriptions.csv',
        path_root='/l/users/maryam.arjemandi/corniche',
        augment_horizontal_flip=True,
        augment_vertical_flip=True
    )
    print("Loading Data")
    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=8, 
        pin_memory=True
    ) 

    # ------------ Initialize Model ------------
    print("Data Loaded")
    model = VAE(
        in_channels=3, 
        out_channels=3, 
        emb_channels=8,
        spatial_dims=2,
        hid_chs =    [64, 128, 256,  512], 
        kernel_sizes=[3,  3,   3,    3],
        strides =    [1,  2,   2,    2],
        deep_supervision=1,
        use_attention='none',
        loss=torch.nn.MSELoss,
        embedding_loss_weight=1e-6
    )
    print("VAE Initialized")
    
    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"  
    min_max = "min"
    save_and_sample_every = 50
    
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=30,
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    
    # Checkpoint path for pre-trained model
    checkpoint_path = '/home/maryam.arjemandi/Documents/medfusion/runs/2024_10_14_210632/last.ckpt'  # Adjust to your actual checkpoint path

    print("Trainer Initialized")
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        limit_val_batches=0,  # No validation
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
        resume_from_checkpoint=checkpoint_path  # Resume from the specified checkpoint
    )
    
    print("Starting Training")
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    torch.save(model.state_dict(), str(path_run_dir / 'best_latent_embedder_model.pth'))
