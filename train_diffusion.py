from email.mime import audio
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
import torchio as tio

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import LabelEmbedder, TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE

from medical_diffusion.data.datasets.dataset_simple_2d import SimpleDataset2D, UltrasoundDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

if __name__ == "__main__":
    # ------------ Load Data ----------------
    dataset = UltrasoundDataset(
        csv_file='/home/maryam.arjemandi/Desktop/Updated_Paraphrased_Ultrasound_Descriptions.csv',  # CSV file containing 'Image_name' and 'Description'
        path_root='/l/users/maryam.arjemandi/corniche'  # Directory containing images
    )
    
    # Use SimpleDataModule with your dataset
    dm = SimpleDataModule(
        ds_train=dataset,
        batch_size=16,  # Adjust batch size if necessary
        pin_memory=True,
    )
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)

    # ------------ Initialize Model ------------
    cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'emb_dim': 1024,
        'num_classes': 2  # You might want to adjust this according to your dataset labels
    }

    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024  # Adjust based on your model architecture
    }

    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': 8, 
        'out_ch': 8, 
        'spatial_dims': 2,
        'hid_chs': [256, 256, 512, 1024],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [1, 2, 2, 2],
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder': cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block': True,
        'use_attention': 'none',
    }

    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002,
        'beta_end': 0.02,
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space ------------
    latent_embedder = VAE
    latent_embedder_checkpoint = '/home/maryam.arjemandi/Documents/medfusion/runs/2024_10_16_223153/last.ckpt'

    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint=latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.5,  # Set to 0 during training to disable
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=1000
    )

    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"
    min_max = "min"
    save_and_sample_every = 100

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
        save_top_k=2,
        mode=min_max,
    )
    
    # Checkpoint path for pre-trained model
    checkpoint_path = '/home/maryam.arjemandi/Documents/medfusion/runs/2024_10_21_003712/last.ckpt'  # Adjust to your actual checkpoint path

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
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
