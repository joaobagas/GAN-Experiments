import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from generator import Generator
from discriminator import Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def set_nets(device, ngpu):

    # Create the Generator and discriminators
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # Handle multi-GPU
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    netG.apply(weights_init)
    netD.apply(weights_init)    

    return netG, netD

def train(netD, netG, dataloader, device, lr = 0.0002, epochs = 5, beta1 = 0.5):

    # Setup optimizers for the generator and the discriminator
    optimD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Initialize the criterion
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label, fake_label = 1., 0.

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):

            ###########################
            # Train the Discriminator #
            ###########################

            #-------------#
            # Real Images #
            #-------------#

            # Set the gradients to zero
            netD.zero_grad()

            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through Discriminator
            output = netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for Discriminator in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            #-------------#
            # Fake Images #
            #-------------#

            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)

            # Generate fake image batch with Generator
            fake = netG(noise)
            label.fill_(fake_label)

            # Classify all fake batch with Discriminator
            output = netD(fake.detach()).view(-1)

            # Calculate Discriminator's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of the discriminator for the real and fake images
            errD = errD_real + errD_fake

            # Update the discriminator
            optimD.step()

            ###########################
            #   Train the Generator   #
            ###########################

            # Set the gradients to zero
            netG.zero_grad()

            # Fake labels are real for generator cost
            label.fill_(real_label) 

            # Since we updated the discriminator, perform another forward pass of all-fake batch through discriminator
            output = netD(fake).view(-1)

            # Calculate generator's loss based on this output
            errG = criterion(output, label)

            # Calculate gradients for generator
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update the generator
            optimG.step()
            
            ###########################
            #     Writing results     #
            ###########################

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1