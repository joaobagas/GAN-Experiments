device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 3e-4
z_dim = 64
image_dim = 28*28*1 
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)

tranforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset = datasets.MNIST(root='dataset/', transform = transforms, download = True)

loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter("runs/GAN_MNIST/fake")
writer_real = SummaryWriter("runs/GAN_MNIST/real")

step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real.view(-1))
        lossD_real = criterion(disc_real, torch.ones_like)
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zerosLike(disc_fake))
        lossD = (lossD_real - lossD_fake)/2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()