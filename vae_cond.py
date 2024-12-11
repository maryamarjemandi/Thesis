import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from medical_diffusion.data.datasets.dataset_simple_2d import SimpleDataset2D
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.model_base import BasicModel
from medical_diffusion.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from medical_diffusion.models.embedders.latent_embedders import DiagonalGaussianDistribution
from lpips import LPIPS
from torchvision import transforms as T
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch.nn as nn

os.environ.pop("SLURM_NTASKS", None)
os.environ.pop("SLURM_NTASKS_PER_NODE", None)
os.environ.pop("SLURM_JOB_ID", None)
os.environ.pop("SLURM_JOB_NODELIST", None)
torch.cuda.empty_cache()


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
    def __init__(self, csv_file, path_root, augment_horizontal_flip=False, *args, **kwargs):
        super().__init__(path_root, *args, **kwargs)
        self.labels = pd.read_csv(csv_file)
        self.item_pointers = self.labels['Image Name'].tolist()
        self.locations = self.labels['Location'].tolist()

        transform_list = [T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
        if augment_horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip())
        
        self.transform = T.Compose(transform_list)

    def __getitem__(self, index):
        image_name = self.item_pointers[index]
        location_name = self.locations[index]
        
        one_hot_vector = torch.tensor(location_to_one_hot[location_name], dtype=torch.float32)
        
        path_item = Path(self.path_root) / image_name
        img = self.load_item(path_item)
        
        if img is None:
            
            return self.__getitem__((index + 1) % len(self.item_pointers))
        
        img = self.transform(img)
        return {'uid': image_name, 'image': img, 'condition': one_hot_vector}
    
    def load_item(self, path_item):
        try:
            return Image.open(path_item).convert('RGB')
        except UnidentifiedImageError:
            print(f"Warning: Could not open {path_item}. Skipping this file.")
            return None

class VAE(BasicModel):
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        spatial_dims=2,
        emb_channels=4,
        num_structures=15,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        norm_name=("GROUP", {'num_groups':8, "affine": True}),
        act_name=("Swish", {}),
        dropout=None,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        embedding_loss_weight=1e-6,
        perceiver=LPIPS,
        perceiver_kwargs={},
        perceptual_loss_weight=1.0,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler=None,
        lr_scheduler_kwargs={},
        loss=torch.nn.L1Loss,
        loss_kwargs={'reduction': 'none'},
        sample_every_n_steps=1000
    ):
        
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )

        
        self.sample_every_n_steps = sample_every_n_steps
        self.embedding_loss_weight = embedding_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.num_structures = num_structures
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None
        
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        
        
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels + num_structures,
            hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=act_name, 
            norm_name=norm_name,
            emb_channels=None
        )

        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims=spatial_dims, 
                in_channels=hid_chs[i-1], 
                out_channels=hid_chs[i], 
                kernel_size=kernel_sizes[i], 
                stride=strides[i],
                downsample_kernel_size=kernel_sizes[i],
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None
            )
            for i in range(1, self.depth)
        ])

        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2*emb_channels, 3),
            BasicBlock(spatial_dims, 2*emb_channels, 2*emb_channels, 1)
        )

        self.quantizer = DiagonalGaussianDistribution()

        
        self.inc_dec = ConvBlock(
            spatial_dims, 
            emb_channels + num_structures,
            hid_chs[-1], 
            3, 
            act_name=act_name, 
            norm_name=norm_name
        )

        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims=spatial_dims, 
                in_channels=hid_chs[i+1], 
                out_channels=hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=strides[i+1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth-1)
        ])

        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision+1)
        ])

        self.loss_fct = loss(**loss_kwargs)
        
        
        self.save_hyperparameters()

   

    def encode(self, x, structure_onehot):
        x_combined = torch.cat([x, structure_onehot], dim=1)
        h = self.inc(x_combined)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z

    def decode(self, z, structure_onehot):
        z_combined = torch.cat([z, structure_onehot], dim=1)
        h = self.inc_dec(z_combined)
        for i in range(len(self.decoders)-1, -1, -1):
            h = self.decoders[i](h)
        x = self.outc(h)
        return x

    def prepare_condition(self, structure_label, spatial_size, target_size=None):
        
        structure_onehot = structure_label.float()  
        structure_onehot = structure_onehot.view(structure_onehot.size(0), -1, 1, 1)
        
        if target_size is not None:
            spatial_size = target_size
            
        return structure_onehot.expand(-1, -1, spatial_size[0], spatial_size[1])

    def forward(self, batch):
        x_in = batch['image']
        structure_label = batch['condition']
        
        
        structure_onehot = self.prepare_condition(structure_label, x_in.shape[2:])
        h = self.inc(torch.cat([x_in, structure_onehot], dim=1))
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        
        
        z_q, emb_loss = self.quantizer(z)
        
        
        z_spatial_size = z_q.shape[2:]  
        structure_onehot_latent = self.prepare_condition(structure_label, spatial_size=None, target_size=z_spatial_size)
        
        
        z_combined = torch.cat([z_q, structure_onehot_latent], dim=1)
        out_hor = []
        h = self.inc_dec(z_combined)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None 
            h = self.decoders[i](h)
        out = self.outc(h)

        return out, out_hor[::-1], emb_loss
        
    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        
        x = batch['image']
        target = x
        
        
        pred, pred_vertical, emb_loss = self(batch)
        
        
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss * self.embedding_loss_weight
        
        
        with torch.no_grad():
            logging_dict = {
                'loss': loss,
                'emb_loss': emb_loss,
                'L2': torch.nn.functional.mse_loss(pred, target),
                'L1': torch.nn.functional.l1_loss(pred, target),
                'ssim': ssim_metric(
                    ((pred + 1) / 2).clamp(0, 1),
                    ((target.type(pred.dtype) + 1) / 2),
                    data_range=1.0)}
        
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], 
                    on_step=True, on_epoch=True)
        
        
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir) / 'images'
            path_out.mkdir(parents=True, exist_ok=True)
            
            def depth2batch(image):
                return (image if image.ndim < 5 else torch.swapaxes(image[0], 0, 1))
            
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)])
            save_image(images, path_out/f'sample_{log_step}.png', 
                      nrow=x.shape[0], normalize=True)
        
        return loss

    def rec_loss(self, pred, pred_vertical, target):
        # Reconstruction loss
        loss = 0
        rec_loss = (self.loss_fct(pred, target) + 
                   self.perception_loss(pred, target) + 
                   self.ssim_loss(pred, target))
        loss += torch.sum(rec_loss) / pred.shape[0]

        # Handle deep supervision if enabled
        for i, pred_i in enumerate(pred_vertical):
            target_i = torch.nn.functional.interpolate(
                target, 
                size=pred_i.shape[2:], 
                mode='nearest-exact', 
                align_corners=None
            )
            rec_loss_i = (self.loss_fct(pred_i, target_i) + 
                         self.perception_loss(pred_i, target_i) + 
                         self.ssim_loss(pred_i, target_i))
            loss += torch.sum(rec_loss_i) / pred.shape[0]

        return loss

    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth < 2):
            self.perceiver.eval()
            return self.perceiver(pred, target) * self.perceptual_loss_weight
        return 0

    def ssim_loss(self, pred, target):
        return 1 - ssim_metric(
            ((pred + 1) / 2).clamp(0, 1),
            ((target.type(pred.dtype) + 1) / 2),
            data_range=1.0)

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)

    

    ds = UltrasoundDataset(
        csv_file='/home/maryam.arjemandi/Documents/medfusion/train_images.csv',
        path_root='/l/users/maryam.arjemandi/New_train_flat/',
        augment_horizontal_flip=True
    )
    
    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=4, 
        pin_memory=True
    ) 

    model = VAE()
    
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
        save_on_train_epoch_end=True
    )
    checkpoint_path = '/home/maryam.arjemandi/Documents/medfusion/runs/2024_12_01_040025/last.ckpt'
    
    print("Trainer Initialized")
    trainer = Trainer(
        precision=16,
        accelerator='gpu',
        devices=[0] if torch.cuda.is_available() else None,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every,
        limit_val_batches=0,  
        min_epochs=1,  
        max_epochs=100,
        num_sanity_val_steps=2,
        resume_from_checkpoint=checkpoint_path
     
   )
    trainer.fit(model, datamodule=dm)
    torch.save(model.state_dict(), str(path_run_dir / 'best_latent_embedder_model.pth'))
