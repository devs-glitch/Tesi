'''Take clean data and transform in tensors'''

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define transformations
tranformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

# Initialize Imagefolder
data_dir = 'data/processed/'
my_dataset = datasets.ImageFolder(root=data_dir, transform=tranformations)

# Initialize DataLoader
my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

# Test
if __name__ == '__main__':
    images, tags = next(iter(my_dataloader))

    print(images.shape)
    print(tags.shape)
