import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from functions import *

# Decide which device we want to run on
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Root directory for dataset
dataroot = "/Users/bernardo.dias/Desktop/Documentos/Projects/GANs/dataset"

ctransforms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                                      transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = dset.ImageFolder(root=dataroot, transform=ctransforms)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) 

# Set up the neural networks
netD, netG = set_nets(device, ngpu)

# Run the training function
train(netD, netG, dataloader, device, lr=0.0002, epochs = 5, beta1=0.5)