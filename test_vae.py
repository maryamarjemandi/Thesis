from pathlib import Path
import torch
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
from torchvision import transforms as T
from torchvision.transforms.functional import rgb_to_grayscale
from torch.nn.functional import conv2d
from medical_diffusion.data.datasets.dataset_simple_2d import SimpleDataset2D
from medical_diffusion.data.datamodules import SimpleDataModule
from vae_cond import VAE


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

# Dataset class
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
        img = self.transform(img)
        return {'uid': image_name, 'image': img, 'condition': one_hot_vector}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB')


def gaussian_kernel(channels, kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    g = g / g.sum()
    g_2d = g[:, None] * g[None, :]
    g_2d = g_2d.expand(channels, 1, kernel_size, kernel_size)
    return g_2d

def apply_smoothing(image, kernel):
    if image.dim() == 3:
        image = image.unsqueeze(0)  
    smoothed = conv2d(image, kernel, padding=kernel.size(-1) // 2, groups=image.size(1))
    return smoothed.squeeze(0)  

if __name__ == "__main__":
    checkpoint_path = '/home/maryam.arjemandi/Documents/medfusion/runs/2024_12_11_000125/last.ckpt'
    path_run_dir = Path.cwd() / 'train_results'
    path_run_dir.mkdir(parents=True, exist_ok=True)

    model = VAE.load_from_checkpoint(checkpoint_path)
    model.eval()

    ds = UltrasoundDataset(
        csv_file='/home/maryam.arjemandi/Documents/medfusion/test_images.csv',
        path_root='/l/users/maryam.arjemandi/New_test_flat/',
        augment_horizontal_flip=False
    )
    dm = SimpleDataModule(ds_train=ds, batch_size=4, pin_memory=True)

    output_dir = path_run_dir / 'reconstructions'
    output_dir.mkdir(exist_ok=True)

    
    channels = 1  
    kernel_size = 5
    sigma = 1.0
    gaussian_filter = gaussian_kernel(channels, kernel_size, sigma)

    with torch.no_grad():
        for batch in dm.train_dataloader():
            inputs = batch['image']
            conditions = batch['condition']
            uids = batch['uid']

            reconstructions, _, _ = model({'image': inputs, 'condition': conditions})
            reconstructions_gray = rgb_to_grayscale(reconstructions)

            for i, (input_img, recon_img, uid) in enumerate(zip(inputs, reconstructions_gray, uids)):
                input_img_gray = rgb_to_grayscale(input_img)

                
                smoothed_recon_img = apply_smoothing(recon_img, gaussian_filter)

                combined = torch.cat((input_img_gray, smoothed_recon_img), dim=2)  
                save_image(combined, output_dir / f'{uid}_comparison.png')

    print(f"Reconstructed images with smoothing saved in {output_dir}")
