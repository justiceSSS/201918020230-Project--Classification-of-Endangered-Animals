from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils

# Define the image transformations
image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load a sample image
img = Image.open('./data/train/panda/panda (90).jpg')

# Apply the image transformations
img_transformed = image_transforms(img)

# Create a grid of transformed images
grid = utils.make_grid(img_transformed)

# Plot the grid
plt.imshow(grid.permute(1, 2, 0))
plt.show()
