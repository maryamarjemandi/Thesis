import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

class SimpleDataset2D:
    def __init__(self, path_root, item_pointers=None, transform=None):
        self.path_root = Path(path_root)
        self.item_pointers = item_pointers or []
        self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        image_name = self.item_pointers[index]
        path_item = self.path_root / f"{image_name}.png"
        img = self.load_item(path_item)
        
        if self.transform:
            img = self.transform(img)
        
        return {'uid': image_name, 'source': img}

    def load_item(self, path_item):
        # Load the image using PIL
        return Image.open(path_item).convert('RGB')

# Define UltrasoundDataset as a subclass of SimpleDataset2D
class UltrasoundDataset(SimpleDataset2D):
    def __init__(self, csv_file, path_root, *args, **kwargs):
        super().__init__(path_root, *args, **kwargs)
        self.labels = pd.read_csv(csv_file)
        
        # Use the 'Image_name' column for image paths and 'Description' for labels
        self.item_pointers = self.labels['Image_name'].tolist()
        self.descriptions = self.labels['Description'].tolist()
        
        if self.transform is None:
            self.transform = T.Compose([
                
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5)  # Normalization to [-1, 1]
            ])

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