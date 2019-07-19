from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from torchvision.utils import save_image

if not os.path.exists('./img'):
    os.mkdir('./img')

# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "data/celeba"        # Root directory for dataset
workers = 2                     # Number of workers for dataloader
batch_size = 128                # Batch size during training
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),       # image_size = 64
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 三个图像通道正则化
                           ]))
# Create the dataloader (num_workers = 2, batch_size = 128)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


'''
1. the weigth initialization strategy
2. generator
3. discriminator
4. loss function、optimizer
5. training loop
6. save model
'''

# the weigth initialization strategy
# DCGAN:  all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02.
def weights_init(m):
    classname = m.__class__.__name__
    # 卷积层+ 反卷积层 convolutional, convolutional-transpose
    #  batch normalization layers
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)   #self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

'''
Xinda：
class类包含：
    类的属性：类中所涉及的变量 
    类的方法：类中函数

_init_函数（方法）
    1. 带有两个下划线开头的函数是声明该属性为私有,不能在类地外部被使用或直接访问
    2. init函数（方法）支持带参数的类的初始化 ，也可为声明该类的属性 
    3. init函数（方法）的第一个参数必须是 self（self为习惯用法，也可以用别的名字），后续参数则可 以自由指定，和定义函数没有任何区别。

super()
在类的继承里面super()非常常用， 它解决了子类调用父类方法的一些问题， 父类多次被调用时只执行一次， 优化了执行逻辑

class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
'''



# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()    # super 用于继承父类，init()用于
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nz - length of latent vector
            # ngf relates to the depth of feature maps carried through the generator
            # ndf - sets the depth of feature maps propagated through the discriminator
            # nn.ConvTranspose2d(in_channels=100, out_channels=64*8, kernel_size=4, stride=1, padding=0)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),                                  # 与output_channel 相同
            nn.ReLU(True),
            # https://www.zhihu.com/question/48279880 :教你如何计算转置卷积，输入的大小 与输出的大小之间的关系
            # o = s(i-1)-2p +k
            # feature map size: s(i-1)-2p +k = 1x(1-1)-2x0 +4 = 4
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # o = 2(4-1)-2x1 +4 =8
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # o = 2(8-1)-2x1 +4 =16
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # # o = 2(16-1)-2x1 +4 =32
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
print(netG)

# The DCGAN paper mentions it is a good practice to use strided convolution
# rather than pooling to downsample because it lets the network learn its own pooling function.


class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc = 3) x 64 x 64 | nc = the number of channel
            # ndf =  Size of feature maps in discriminator
            # nn.Conv2d(nc=3, ndf=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # inplace-选择是否进行覆盖运算
            # 对输入的每一个元素运用$f(x) = max(0, x) + {negative_slope} * min(0, x)$
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create Discriminator and to device
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)
print(netD)

# Loss Functions and Optimizers

# Initialize BCELoss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

# torch.rand(*sizes, out=None) → Tensor;
# 返回一个张量，包含了从区间[0,1)的均匀分布中抽取的一组随机数，形状由可变参数sizes 定义
# (64, 100, 1, 1): 第一个纬度=64 表示每一批数量为64，是训练时候的128的一半，这个不影响Generator的参数
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


##################################################### Training##########################################################

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):                                             # num_epochs = 5
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):                                # i = 0~1583 , start = 0

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        #################################### Train with all-real batch #################################################
        netD.zero_grad()
        # Format batch
        # data is a list,
        # data[0]=the value of batch images
        # data[1]=the label of batch images
        # real_cpu :torch.Size([128, 3, 64, 64])
        real_cpu = data[0].to(device)                   # 批图像的数据
        b_size = real_cpu.size(0)                        # real_cpu.size(0) = batch_size =128
        # torch.full(size, fill_value)
        # 这个有时候比较方便，把fill_value这个数字变成size形状的张量
        label = torch.full((b_size,), real_label, device=device)        # torch.Size([128])，现在变成了全1

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)            # 变成一维
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # Use torch.Tensor.item() to get a Python number from a tensor containing a single value:
        D_x = output.mean().item()

        ################################# Train with all-fake batch   ##################################################
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)   # (128, 100, 1, 1 )
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        ########################################### Output training stats   ############################################
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                fake_image = fake.data
                save_image(fake_image, './img/fake_images-{}.png'.format(epoch + 1))
            # 没有输出去使用，所以保存的图片会跟官网不太一样
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

torch.save(netG.state_dict(), './generator.pth')
torch.save(netD.state_dict(), './discriminator.pth')