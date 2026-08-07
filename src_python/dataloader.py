'''Take clean data and transform in tensors'''

import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import UnidentifiedImageError

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError) as e:

            # corrupted image method
            
            print(f"Skipped corrupted image: {self.samples[index][0]}")
            new_index = random.randint(0, len(self) - 1)
            return self.__getitem__(new_index)

# Define transformations
tranformations = transforms.Compose([
    transforms.CenterCrop((470, 550)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Initialize Imagefolder
data_dir = 'data/processed/'
my_dataset = SafeImageFolder(root=data_dir, transform=tranformations)

'''
The split
'''
total_size = len(my_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_ds, val_ds, test_ds = random_split(my_dataset, [train_size, val_size, test_size], generator=generator)

train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=False)

if __name__ == '__main__':
    print(f"Total # images: {total_size}")
    print(f"Train: {train_size} | Validation: {val_size} | Test: {test_size}")

    images, tags = next(iter(train_dataloader))

    print(f"Dimension batch images: {images.shape}")
    print(f"Dimension batch tags: {tags.shape}")