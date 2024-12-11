import torch
from torchvision.utils import save_image
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
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

def save_histogram(image_tensor, save_path):
    # Convert tensor to numpy array
    image_np = image_tensor.squeeze().numpy()
    plt.figure()
    plt.hist(image_np.ravel(), bins=256, color='gray', alpha=0.75)
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    target_image_names = [
        "MBZUAI-US--006119_Study-US-20171025T114921-1386_Series-1_Image-6.png",
        "MBZUAI-US--006169_Study-US-20181109T113325-2282_Series-1_Image-39.png",
        "MBZUAI-US--006177_Study-US-20100119T084224-3619_Series-1_Image-52.png",
        "MBZUAI-US--006196_Study-US-20181211T134352-3265_Series-1_Image-13.png",
        "MBZUAI-US--006209_Study-US-20170206T110840-3205_Series-1_Image-27.png",
        "MBZUAI-US--006212_Study-US-20150405T081002-2923_Series-1_Image-26.png",
        "MBZUAI-US--006212_Study-US-20150405T081002-2923_Series-1_Image-41.png",
        "MBZUAI-US--006221_Study-US-20140806T153024-9078_Series-1_Image-24.png",
        "MBZUAI-US--006225_Study-US-20140327T114931-1750_Series-1_Image-33.png",
        "MBZUAI-US--006225_Study-US-20140327T114931-1750_Series-1_Image-41.png",
        "MBZUAI-US--006225_Study-US-20140327T114931-1750_Series-1_Image-48.png"
    ]
    
    base_output_dir = Path("/l/users/maryam.arjemandi/Results/")
    ds = UltrasoundDataset(
        csv_file='/l/users/maryam.arjemandi/RecentCornicheFetalUS/LabeledImages/output_images.csv',
        path_root='/l/users/maryam.arjemandi/RecentCornicheFetalUS/LabeledImages/all_test/'
    )

    model = VAE.load_from_checkpoint('/home/maryam.arjemandi/Documents/medfusion/runs/2024_10_16_223153/last.ckpt')
    model.eval()

    multipliers = [1.5, 2, 3]

    with torch.no_grad():
        for i in range(len(ds)):
            batch = ds[i]
            if batch['uid'] not in target_image_names:
                continue
            
            image_name = batch['uid']
            input_img = batch['source'].unsqueeze(0)  # Add batch dimension
            z = model.encode(input_img)

            # Create main folder for each image
            image_output_dir = base_output_dir / image_name
            image_output_dir.mkdir(parents=True, exist_ok=True)

            # Apply each multiplier to each individual dimension
            for multiplier in multipliers:
                for dim in range(z.size(1)):  # Assuming z is (batch_size, 8)
                    modified_z = z.clone()  # Copy to avoid altering the original
                    modified_z[:, dim] *= multiplier  # Apply multiplier only to the current dimension
                    
                    modified_output = model.decode(modified_z)
                    input_denorm = (input_img * 0.5) + 0.5
                    output_denorm = (modified_output * 0.5) + 0.5
                    
                    # Create subfolder for the current dimension
                    dim_output_dir = image_output_dir / f"dim{dim}"
                    dim_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    combined_img = torch.cat((input_denorm.squeeze(), output_denorm.squeeze()), dim=2)
                    img_save_path = dim_output_dir / f'{image_name}_dim{dim}_multiplier_{multiplier}.png'
                    save_image(combined_img, img_save_path)

                    # Save histogram for generated image
                    histogram_save_path = dim_output_dir / f'{image_name}_dim{dim}_multiplier_{multiplier}_histogram.png'
                    save_histogram(output_denorm, histogram_save_path)

            # Apply each multiplier to all dimensions simultaneously
            for multiplier in multipliers:
                modified_z = z.clone() * multiplier  # Apply multiplier to all dimensions
                modified_output = model.decode(modified_z)
                
                input_denorm = (input_img * 0.5) + 0.5
                output_denorm = (modified_output * 0.5) + 0.5
                
                # Create 'alldimensions' subfolder
                all_dims_output_dir = image_output_dir / "alldimensions"
                all_dims_output_dir.mkdir(parents=True, exist_ok=True)
                
                combined_img = torch.cat((input_denorm.squeeze(), output_denorm.squeeze()), dim=2)
                img_save_path = all_dims_output_dir / f'{image_name}_alldimensions_multiplier_{multiplier}.png'
                save_image(combined_img, img_save_path)

                # Save histogram for the "all dimensions" modified image
                histogram_save_path = all_dims_output_dir / f'{image_name}_alldimensions_multiplier_{multiplier}_histogram.png'
                save_histogram(output_denorm, histogram_save_path)

    print("Images with modified z dimensions and histograms saved successfully.")
