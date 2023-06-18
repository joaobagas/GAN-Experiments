# Import from other modules
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Import from other files
from generator import Generator
from discriminator import Discriminator
from functions import train

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dim = 28*28*1
    z_dim = 64

    # Initialize the generator and the discriminator
    dis = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)

    # Get the dataset and initialize the dataloader
    dataset = datasets.MNIST(root='../dataset/', transform = transforms.ToTensor(), download = True)

    # Start the training process with the default values
    train(dis, gen, dataset, device)