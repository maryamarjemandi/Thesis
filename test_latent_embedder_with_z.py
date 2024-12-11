import torch
from torchvision.utils import save_image
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image
from torchvision import transforms as T
from medical_diffusion.data.datasets.dataset_simple_2d import SimpleDataset2D
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.embedders.latent_embedders import VAE

class UltrasoundDataset(SimpleDataset2D):
    def __init__(self, csv_file, path_root, augment_horizontal_flip=False, augment_vertical_flip=False, *args, **kwargs):
        super().__init__(path_root, *args, **kwargs)
        self.labels = pd.read_csv(csv_file)
        self.item_pointers = self.labels['Image_Name'].tolist()
        self.descriptions = self.labels['Description'].tolist()
        transform_list = [
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ]
        if augment_horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip())
        if augment_vertical_flip:
            transform_list.append(T.RandomVerticalFlip())
        self.transform = T.Compose(transform_list)

    def __getitem__(self, index):
        image_name = self.item_pointers[index]
        description = self.descriptions[index]
        path_item = Path(self.path_root) / f"{image_name}"
        img = self.load_item(path_item)
        img = self.transform(img)
        return {'uid': image_name, 'source': img, 'description': description}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB')

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / f'z_variation_{current_time}'
    path_run_dir.mkdir(parents=True, exist_ok=True)

    ds = UltrasoundDataset(
        csv_file='/l/users/maryam.arjemandi/RecentCornicheFetalUS/LabeledImages/output_images.csv',
        path_root='/l/users/maryam.arjemandi/RecentCornicheFetalUS/LabeledImages/all_test/',
        augment_horizontal_flip=True,
        augment_vertical_flip=True
    )
    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=8, 
        pin_memory=True
    ) 

    model = VAE.load_from_checkpoint('/home/maryam.arjemandi/Documents/medfusion/runs/2024_10_16_223153/last.ckpt')
    model.eval()

    z_offsets = [-3, 0, 3]
    
    output_dir = path_run_dir
    output_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dm.train_dataloader()):
            inputs = batch['source']
            z = model.encode(inputs)
            print("Original z:", z)
            for offset in z_offsets:
                modified_z = z + offset
                modified_outputs = model.decode(modified_z)
                inputs_denorm = (inputs * 0.5) + 0.5
                modified_outputs_denorm = (modified_outputs * 0.5) + 0.5
                image_names = batch['uid']
                for j, (input_img, output_img, img_name) in enumerate(zip(inputs_denorm, modified_outputs_denorm, image_names)):
                    combined_img = torch.cat((input_img, output_img), dim=2)
                    save_image(combined_img, output_dir / f'{img_name}_z_offset_{offset}.png')

    print("Images with modified z saved successfully.")
