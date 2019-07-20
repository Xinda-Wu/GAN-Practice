import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

'''
吴鑫达：
    GANs-->WGAN：
    1. 判别器最后一层去掉了sigmoid函数
    2. 生成器和判别器的loss都不取log
    3. 每次跟新判别器的参数之后把它们的绝对值截断到一个不超过一个固定常数c 
    4. 不建议使用基于动量的算法，包括momentum和Adam，推荐使用RMSProp，SGD
'''

batch_size = 128
ngpu = 1                        # Number of GPUs available. Use 0 for CPU mode.


##################################################### 交互式的参数 set parameters ###################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# default （1，28，28） 28x28 = 784
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),                       # range [0, 255] -> [0.0,1.0]
    transforms.Normalize([0.5], [0.5])           # 归一化到 [0.0,1.0],  channel=（channel-mean）/std
])

# MNIST dataset：准备数据
mnist = datasets.MNIST(
    root='./data/MNIST/',
    train=True,
    transform=img_transform,
    download=False
)

# dataloader
dataloader = torch.utils.data.DataLoader(
    dataset=mnist,
    batch_size=batch_size,
    shuffle=True
)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),           # np.prod()函数用来计算所有元素的乘积
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
#  initial Tensor for cuda
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------------------------------------------------------------------------------------------------
# Training
# 1. Train Discriminator(1.loss without log,   2.clip weight)
# 2. Train Generator
# ----------------------------------------------------------------------------------------------------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input(128,1,28,28)
        real_imgs = Variable(imgs.type(Tensor))

        # --------------------------------------------- Train Discriminator -------------------------------------------#
        optimizer_D.zero_grad()

        # Sample noise as generator input | z: torch.Size([128, 100])
        # 高斯分布（Gaussian Distribution）
        # numpy.random.normal(loc=0.0, scale=1.0, size=None)
        #loc=float :此概率分布的均值（对应着整个分布的中心centre）
        # scale=float: 此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
        # size：int or tuple of ints 输出的shape，默认为None，只输出一个值

        # z: torch.Size([128, 100])
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        # detach就是截断反向传播的梯度流,只更新GAN中的Discriminator，而不更新Generator
        # 当反向传播经过这个node时，梯度就不会从这个node往前面传播
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        # n_critic：number of training steps for discriminator per iter =5
        if i % opt.n_critic == 0:
            # ----------------------------------------- Train Generator -----------------------------------------------#
            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))
            # BP & RMSprop
            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1

torch.save(generator.state_dict(), './generator.pth')
torch.save(discriminator.state_dict(), './discriminator.pth')