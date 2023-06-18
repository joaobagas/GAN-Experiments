import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train(dis, gen, dataset, device = "cpu", lr = 3e-4, z_dim = 64, batch_size = 32, num_epochs = 50):
    

    # Initialize the optimizer and the criterion
    opt_dis = optim.Adam(dis.parameters(), lr = lr)
    opt_gen = optim.Adam(gen.parameters(), lr = lr)
    criterion = nn.BCELoss()   

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = dis(real.view(-1))
            lossD_real = criterion(disc_real, torch.ones_like)
            disc_fake = dis(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zerosLike(disc_fake))
            lossD = (lossD_real - lossD_fake)/2
            dis.zero_grad()
            lossD.backward(retain_graph=True)
            opt_dis.step()

            output = dis(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

    return dis, gen