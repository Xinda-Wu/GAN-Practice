import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os


if not os.path.exists('./img'):
    os.mkdir('./img')

def to_img(x):
    # 正则化逆转
    out = 0.5 * (x+1)
    # 将out限制在0-1之间； clamp表示夹紧的意思
    out = out.clamp(0, 1)
    # 将[128,1,28,28] <- [128, 784] 逆转
    out = out.view(-1, 1, 28, 28)
    return out

batch_size =128
num_epoch = 100
z_dimension = 100


# image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),                       # range [0, 255] -> [0.0,1.0]
    transforms.Normalize([0.5], [0.5])           # 归一化到 [0.0,1.0],  channel=（channel-mean）/std
])

# MNIST dataset：准备数据
mnist = datasets.MNIST(
    root='./data/',
    train=True,
    transform=img_transform,
    download=True
)


# dataloader
dataloader = torch.utils.data.DataLoader(
    dataset=mnist,
    batch_size=batch_size,
    shuffle=True
)


# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.dis(x)
        return x

# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.gen(x)
        return x


D = discriminator()
G = generator()

# CUDA is available?
if torch.cuda.is_available():
    print("CUDA is variable")
    D = D.cuda()
    G = G.cuda()
else:
    print("Cuda is not avariable!")

# Binary coss entropy loss and optimizer  :loss度量使用二分类的交叉熵
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# starting train
for epoch in range(num_epoch):
    #for i, (input, target) in enumerate(train_data):
    for i, (img, _) in enumerate(dataloader):
        # num_img = 128
        num_img = img.size(0)
        # ==========train discriminator=============

        # image.size(): torch.Size([128, 1, 28, 28])
        img = img.view(num_img, -1)                         # 将图片展开乘 1x28x28=784
        real_img = Variable(img).cuda()                     # 将tensor变成Variable放入计算图中
        real_label = Variable(torch.ones(num_img)).cuda()   # 定义真实label为1
        fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的label为0

        # computer loss of true_img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out

        # computer loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()      # (128, 100)
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # Closer to 1 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ==========train generator=============
        # computer the loss of fake image
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        # bp and backward
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.06f}, g_loss:{:.6f}'
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.item(), g_loss.item(),
                real_scores.data.mean(), fake_scores.data.mean())
            )

    if epoch == 0:
        # torch.Tensor.cpu() function,
        # which creates and returns a copy of a tensor or even a list of tensors in the CPU.
        '''
        PyTorch0.4中，.data 仍保留，但建议使用 .detach(),
        
         区别在于 .data 返回和 x 的相同数据 tensor, 但不会加入到x的计算历史里，
         且require s_grad = False, 这样有些时候是不安全的, 因为 x.data 不能被 autograd 追踪求微分 。 
         
         .detach() 返回相同数据的 tensor ,且 requires_grad=False ,
         但能通过 in-place 操作报告给 autograd 在进行反向传播的时候.
        
        '''
        real_images = to_img(real_img.cpu().data)  # 从GPU中取出数据放入CPU内存获得数据
        # 保存图片 save_image
        save_image(real_images, './img/real_images.png')
    # 保存假的照片
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch+1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')









