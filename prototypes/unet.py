# https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
import glob

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out





def generate_random_data(height, width, count):
    x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y)

    return X, Y


def generate_img_and_mask(height, width):
    shape = (height, width)

    triangle_location = get_random_location(*shape)
    circle_location1 = get_random_location(*shape, zoom=0.7)
    circle_location2 = get_random_location(*shape, zoom=0.5)
    mesh_location = get_random_location(*shape)
    square_location = get_random_location(*shape, zoom=0.8)
    plus_location = get_random_location(*shape, zoom=1.2)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    # Create target masks
    masks = np.asarray([
        add_filled_square(np.zeros(shape, dtype=bool), *square_location),
        add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
    ]).astype(np.float32)

    return arr, masks


def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

    return arr


def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))


def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array


def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))


def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr


def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr


def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x-1:x+1,y-s:y+s] = True
    arr[x-s:x+s,y-1:y+1] = True

    return arr


def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])


def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))


def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()


def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def generate_images_and_masks_then_plot():
    # Generate some random images
    input_images, target_masks = generate_random_data(192, 192, count=3)

    for x in [input_images, target_masks]:
        print(x.shape)
        print(x.min(), x.max())

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [x.astype(np.uint8) for x in input_images]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in target_masks]

    # Left: Input image (black and white), Right: Target mask (6ch)
    plot_side_by_side([input_images_rgb, target_masks_rgb])


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


class SimDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.inputs = [f for f in sorted(glob.glob(os.path.join(dir, "*.png")))]
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        array =  cv2.imread(self.inputs[idx], cv2.IMREAD_UNCHANGED)
        image = array[:,:,0:3]
        mask = array[:, :, 3]

        image = cv2.resize(image, (192,192))
        mask = cv2.resize(mask, (192,192), interpolation=cv2.INTER_NEAREST)
        mmask = np.zeros((192,192,1), bool)
        mmask[:,:,0][mask > 0] = True
        # mmask[:, :, 1][mask == 1] = True

        mmask = mmask.astype(np.float32)
        mmask = torch.tensor(mmask).permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)

        return [image, mmask]


def get_data_loaders(train_dir, val_dir):
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    train_set = SimDataset(train_dir, transform = trans)
    val_set = SimDataset(val_dir, transform = trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, train_dir, val_dir, num_epochs=25):
    dataloaders = get_data_loaders(train_dir, val_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)

                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "unet.pt")
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run():
    num_class = 1


    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, train_dir, val_dir, num_epochs=60)
    #
    # model.eval()  # Set model to the evaluation mode
    #
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    # ])
    # # # Create another simulation dataset for test
    # test_dataset = SimDataset(3, transform = trans)
    #
    # test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)
    #
    # # Get the first batch
    # inputs, labels = next(iter(test_loader))
    # inputs = inputs.to(device)
    # labels = labels.to(device)
    #
    # # Predict
    # pred = model(inputs)
    # # The loss functions include the sigmoid function.
    # pred = F.sigmoid(pred)
    # pred = pred.data.cpu().numpy()
    # print(pred.shape)
    #
    # # Change channel-order and make 3 channels for matplot
    # input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]
    #
    # # Map each channel (i.e. class) to each color
    # target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    # pred_rgb = [masks_to_colorimg(x) for x in pred]
    #
    # plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])



DATA_DIR = "/home/quentin/Desktop/unet-data"
# run()
device = "cuda"
model = UNet(1).to(device)
model.load_state_dict(torch.load("unet.pt"))
model.eval()

im = cv2.imread("/home/quentin/Desktop/unet-data/val/pitfall_0.25_2023-07-16_10_C_5m024_0021_261_2652_0005.png", cv2.IMREAD_UNCHANGED)
for f in glob.glob("/home/quentin/Desktop/unet-data/val/*.png"):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if im.shape[0] < 500:
        continue
    im = cv2.resize(im, (192,192))
    im_a = im[:,:,0:3]
    mask = im[:,:,3]

    im = im[:,:,0:3]

    im = torch.tensor(im).permute(2, 0, 1)
    im = torch.unsqueeze(im,0)

    im = im.float()
    im = im.to(device).contiguous()

    o = model(im)
    o2 = F.sigmoid(o)

    import matplotlib.pyplot as plt

    print (o.shape)
    # a= o[0,1,:]
    b= o[0,0,:].detach().cpu()
    print(b)
    c= np.copy(b)
    c[c < 50] = 0
    # a = a.detach().cpu()
    # b[b > 0] = 0

    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(im_a, cmap='hot', interpolation='nearest')
    axarr[1].imshow(mask, cmap='hot', interpolation='nearest')
    axarr[2].imshow(b, cmap='hot', interpolation='nearest')
    axarr[3].imshow(c, cmap='hot', interpolation='nearest')
    plt.show()

    # with torch.no_grad:
