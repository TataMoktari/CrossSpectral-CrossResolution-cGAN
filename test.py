
#!/usr/bin/env python

import argparse
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import Generator, Discriminator, FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='folder', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='/media/moktari/External/TBIOM/Cross_Resolution_Cross_Spectral_GAN/CrossResCrossDomain_Moktari/CrossResCrossDomain/data/test', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=512, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='/media/moktari/External/TBIOM/Cross_Resolution_Cross_Spectral_GAN/CrossResCrossDomain_Moktari/CrossResCrossDomain/checkpoints_LNIR_HNIR/generator_final.pth', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='/media/moktari/External/TBIOM/Cross_Resolution_Cross_Spectral_GAN/CrossResCrossDomain_Moktari/CrossResCrossDomain/checkpoints_LNIR_HNIR/discriminator_final.pth', help="path to discriminator weights (to continue training)")

opt = parser.parse_args()
print(opt)

try:
    os.makedirs('output/high_res_fake')
    os.makedirs('output/high_res_real')
    os.makedirs('output/low_res')
except OSError:
    pass


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    print("Let's use", torch.cuda.device_count(), "GPUs!")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.upSampling),
                                transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Scale(int(opt.imageSize/2)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

# Equivalent to un-normalizing ImageNet (for correct visualization)
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

if opt.dataset == 'folder':
    # folder dataset
    dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
elif opt.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root=opt.dataroot, download=True, train=False, transform=transform)
elif opt.dataset == 'cifar100':
    dataset = datasets.CIFAR100(root=opt.dataroot, download=True, train=False, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

generator = Generator(16, opt.upSampling)
if opt.generatorWeights != '':
    generator.load_state_dict(torch.load(opt.generatorWeights))
print(generator)

if torch.cuda.device_count()>1:
    generator = nn.DataParallel(generator, device_ids=[0,1])

target_real = Variable(torch.ones(opt.batchSize,1))
target_fake = Variable(torch.zeros(opt.batchSize,1))

# if gpu is to be used
# if opt.cuda:
generator.cuda()
target_real = target_real.cuda()
target_fake = target_fake.cuda()

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, int(opt.imageSize/2))

print('Test started...')
mean_generator_content_loss = 0.0
mean_generator_adversarial_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

# Set evaluation mode (not training)
generator.eval()


for i, data in enumerate(dataloader):
    # Generate data
    # high_res_real, _ = data
    concat, _ = data

    ##### LR NIR to HR VIS Training
    # high_res_real = concat[:,:, :, 512:1024]
    # low_res_real  = concat[:,:, :, 0:512]

    ##### LR VIS to HR NIR Training
    high_res_real = concat[:, :, :, 0:512]
    low_res_real = concat[:, :, :, 0:512]

    # Downsample images to low resolution
    for j in range(opt.batchSize):
        # LR to HR transformation
        low_res[j] = scale(low_res_real[j])
        #HR VIS to HR NIR transformation
        high_res_real[j] = normalize(high_res_real[j])

    # Generate real and fake inputs
    # if opt.cuda:
    high_res_real = Variable(high_res_real.cuda())
    high_res_fake = generator(low_res)

    # high_res_fake = high_res_fake.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    import  matplotlib.pyplot as plt

    # plt.imshow(high_res_fake)
    # plt.show()
    for j in range(opt.batchSize):
        save_image(unnormalize(high_res_real.data[j]), 'output/high_res_real/' + str(i*opt.batchSize + j) + '.png')
        save_image(unnormalize(high_res_fake.data[j]), 'output/high_res_fake/' + str(i*opt.batchSize + j) + '.png')
        save_image(unnormalize(low_res[j]), 'output/low_res/' + str(i*opt.batchSize + j) + '.png')
