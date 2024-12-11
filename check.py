import torch
from pathlib import Path
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
    target_image_name = "MBZUAI-US--006119_Study-US-20171025T114921-1386_Series-1_Image-6.png"
    target_image_path = Path("/l/users/maryam.arjemandi/RecentCornicheFetalUS/LabeledImages/all_test") / target_image_name

    # Load dataset and data module
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

    # Iterate over batches to find the target image
    with torch.no_grad():
        found = False
        for i, batch in enumerate(dm.train_dataloader()):
            inputs = batch['source']
            z = model.encode(inputs)

            # Iterate through the batch and check each image name
            for j, image_name in enumerate(batch['uid']):
                if image_name == target_image_name:
                    print(f"Found target image: {image_name}")
                    print("z value for the target image:")
                    print(z[j].cpu().numpy())
                    
                    # Save the z value for the target image
                    torch.save(z[j], '/home/maryam.arjemandi/Documents/medfusion/z_value.pt')
                    
                    found = True
                    break

            if found:
                break  # Exit outer loop once the target image is found
        if not found:
            print(f"Image {target_image_name} not found in the dataset.")
