---
title: Convolutional Autoencoders
description: >-
  Use convolutional neural networks for image compression
date: 2020-09-07
categories: [Blog, Tutorial]
tags: [Python, Machine Learning, Deep Learning]
pin: true
author: ks
---

The success of Convolutional Neural Networks at image classification is well known but the same conceptual backbone can be used for other image tasks as well, for example image compression. In this post, I will use [Convolutional Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) to re-generate compressed images for the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). Let's look at a few random images:

![conv-autoencoder-1.png](/assets/conv-autoencoder-1.png)

## Setup

The FMNIST dataset is available directly through the torchvision package. We can load it directly into our environment using torchvision datasets module:

```python
from torchvision import transforms, datasets

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root=root_dir, train=True,
                                   download=True,
                                   transform=transform)
test_data = datasets.FashionMNIST(root=root_dir, train=False,
                                  download=True,
                                  transform=transform)
```
After defining the datasets, we can define data loaders, which will feed batches of images into our neural network:

```python
from torch.utils.data import DataLoader
num_workers = 0
batch_size = 20

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
```

## Architecture

The architecture consists of two segments:
1. Encoder: Looks similar to the regular convolutional pyramid of CNN's
2. Decoder: Converts the narrow representation to wide, reconstructed image. It applies multiple transpose convolutional layers to go from compressed representation to a regular image

Below I define the two set of layers using PyTorch nn.Module:

### Encoder

During the encoding phase, we pass the images through 2 convolutional layers, each followed by a max pool layer. The final dimension of the encoded image is 4 channels of 7 x 7 matrices.

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 4, (3, 3), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        return x
```

### Decoder

During the decoding phase, we pass the decoded image through transpose convolutional layers to increase the dimensions along width and height while bringing the number of channels down from 4 to 1.

```python
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_conv1 = nn.ConvTranspose2d(4, 16, (2, 2), stride=(2, 2))
        self.t_conv2 = nn.ConvTranspose2d(16, 1, (2, 2), stride=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t_conv1(x)
        x = functional.relu(x)
        x = self.t_conv2(x)
        x = torch.sigmoid(x)
        return x
```

## Full Network

In the full network, we combine both layers where encoder layer feeds into the decoder layer.

```python
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

![conv-autoencoder-2.png](/assets/conv-autoencoder-2.png)

## Training

During the training phase, we pass batches of images to our network. We finally compare the actual re-constructed image with the original image using the MSE Loss function to check final pixel loss. We optimize for minimum loss between original and re-constructed image using the Adam optimizer. We train the model for a few epochs and stop after the loss function doesn't show signs of decreasing further. Here's how the training loop looks like:

```python
model = AutoEncoder()
if use_gpu:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_losses = []
for epoch in range(1, n_epochs + 1):
    logger.info(f"[EPOCH {epoch}: Starting training")
    train_loss = 0.0
    batches = len(train_loader)
    for data, _ in tqdm(train_loader, total=batches):
        optimizer.zero_grad()
        if use_gpu:
            data = data.cuda() # transfer data to gpu
        output = model(data) # calculate predicted value
        loss = criterion(output, data) # calculate loss function
        loss.backward() # back propagation
        optimizer.step() # take an optimizer step
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader) # calculate average loss
    logger.info(
        f"[EPOCH {epoch}: Training loss {np.round(train_loss, 6)}")
    train_losses.append(train_loss)
```

### Training Logs

```
2020-09-07 20:30:13.750 | WARNING  | autoencoders.convolutional_autoencoder:__init__:91 - CUDA not available/enabled. Using CPU
2020-09-07 20:30:13.751 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 1: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.26it/s]
2020-09-07 20:30:23.810 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 1: Training loss 0.437312
2020-09-07 20:30:23.810 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 2: Starting training
100%|██████████| 3000/3000 [00:09<00:00, 301.93it/s]
2020-09-07 20:30:33.747 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 2: Training loss 0.237542
2020-09-07 20:30:33.747 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 3: Starting training
100%|██████████| 3000/3000 [00:09<00:00, 302.29it/s]
2020-09-07 20:30:43.671 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 3: Training loss 0.218295
2020-09-07 20:30:43.672 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 4: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 294.49it/s]
2020-09-07 20:30:53.859 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 4: Training loss 0.207977
2020-09-07 20:30:53.859 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 5: Starting training
100%|██████████| 3000/3000 [00:09<00:00, 300.07it/s]
2020-09-07 20:31:03.857 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 5: Training loss 0.202394
2020-09-07 20:31:03.857 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 6: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.93it/s]
2020-09-07 20:31:13.893 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 6: Training loss 0.199011
2020-09-07 20:31:13.893 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 7: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 284.79it/s]
2020-09-07 20:31:24.428 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 7: Training loss 0.196497
2020-09-07 20:31:24.428 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 8: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 284.40it/s]
2020-09-07 20:31:34.977 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 8: Training loss 0.194811
2020-09-07 20:31:34.977 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 9: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 299.79it/s]
2020-09-07 20:31:44.984 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 9: Training loss 0.193926
2020-09-07 20:31:44.984 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 10: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 294.04it/s]
2020-09-07 20:31:55.187 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 10: Training loss 0.193171
2020-09-07 20:31:55.187 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 11: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.18it/s]
2020-09-07 20:32:05.248 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 11: Training loss 0.192589
2020-09-07 20:32:05.249 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 12: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.21it/s]
2020-09-07 20:32:15.309 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 12: Training loss 0.192103
2020-09-07 20:32:15.309 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 13: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 294.76it/s]
2020-09-07 20:32:25.487 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 13: Training loss 0.191604
2020-09-07 20:32:25.487 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 14: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 296.91it/s]
2020-09-07 20:32:35.592 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 14: Training loss 0.191146
2020-09-07 20:32:35.592 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 15: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.23it/s]
2020-09-07 20:32:45.651 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 15: Training loss 0.190652
2020-09-07 20:32:45.652 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 16: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.84it/s]
2020-09-07 20:32:55.691 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 16: Training loss 0.190262
2020-09-07 20:32:55.691 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 17: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.83it/s]
2020-09-07 20:33:05.764 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 17: Training loss 0.189871
2020-09-07 20:33:05.764 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 18: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.28it/s]
2020-09-07 20:33:15.856 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 18: Training loss 0.189549
2020-09-07 20:33:15.856 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 19: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 295.27it/s]
2020-09-07 20:33:26.017 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 19: Training loss 0.189221
2020-09-07 20:33:26.017 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 20: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 294.82it/s]
2020-09-07 20:33:36.193 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 20: Training loss 0.189024
2020-09-07 20:33:36.193 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 21: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.50it/s]
2020-09-07 20:33:46.277 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 21: Training loss 0.188787
2020-09-07 20:33:46.277 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 22: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.19it/s]
2020-09-07 20:33:56.338 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 22: Training loss 0.188466
2020-09-07 20:33:56.338 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 23: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.14it/s]
2020-09-07 20:34:06.401 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 23: Training loss 0.18817
2020-09-07 20:34:06.401 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 24: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.30it/s]
2020-09-07 20:34:16.492 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 24: Training loss 0.188001
2020-09-07 20:34:16.492 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 25: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 295.21it/s]
2020-09-07 20:34:26.655 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 25: Training loss 0.187785
2020-09-07 20:34:26.655 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 26: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.07it/s]
2020-09-07 20:34:36.720 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 26: Training loss 0.18753
2020-09-07 20:34:36.720 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 27: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 296.21it/s]
2020-09-07 20:34:46.848 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 27: Training loss 0.187273
2020-09-07 20:34:46.849 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 28: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.57it/s]
2020-09-07 20:34:56.930 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 28: Training loss 0.187002
2020-09-07 20:34:56.931 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 29: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.41it/s]
2020-09-07 20:35:07.018 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 29: Training loss 0.186654
2020-09-07 20:35:07.018 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 30: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.83it/s]
2020-09-07 20:35:17.091 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 30: Training loss 0.186408
2020-09-07 20:35:17.091 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 31: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.39it/s]
2020-09-07 20:35:27.179 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 31: Training loss 0.18608
2020-09-07 20:35:27.180 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 32: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 295.95it/s]
2020-09-07 20:35:37.317 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 32: Training loss 0.185741
2020-09-07 20:35:37.317 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 33: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.16it/s]
2020-09-07 20:35:47.413 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 33: Training loss 0.185363
2020-09-07 20:35:47.413 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 34: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.01it/s]
2020-09-07 20:35:57.514 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 34: Training loss 0.184985
2020-09-07 20:35:57.514 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 35: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.53it/s]
2020-09-07 20:36:07.597 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 35: Training loss 0.184487
2020-09-07 20:36:07.597 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 36: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.52it/s]
2020-09-07 20:36:17.681 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 36: Training loss 0.184044
2020-09-07 20:36:17.681 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 37: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 296.39it/s]
2020-09-07 20:36:27.803 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 37: Training loss 0.183633
2020-09-07 20:36:27.803 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 38: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.19it/s]
2020-09-07 20:36:37.898 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 38: Training loss 0.183359
2020-09-07 20:36:37.898 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 39: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 293.79it/s]
2020-09-07 20:36:48.110 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 39: Training loss 0.183028
2020-09-07 20:36:48.110 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 40: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.24it/s]
2020-09-07 20:36:58.203 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 40: Training loss 0.182765
2020-09-07 20:36:58.203 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 41: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 296.77it/s]
2020-09-07 20:37:08.313 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 41: Training loss 0.182514
2020-09-07 20:37:08.313 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 42: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 294.76it/s]
2020-09-07 20:37:18.491 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 42: Training loss 0.182298
2020-09-07 20:37:18.491 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 43: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.70it/s]
2020-09-07 20:37:28.569 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 43: Training loss 0.182133
2020-09-07 20:37:28.569 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 44: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 294.96it/s]
2020-09-07 20:37:38.740 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 44: Training loss 0.181893
2020-09-07 20:37:38.740 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 45: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 291.89it/s]
2020-09-07 20:37:49.018 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 45: Training loss 0.181797
2020-09-07 20:37:49.019 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 46: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.31it/s]
2020-09-07 20:37:59.109 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 46: Training loss 0.181623
2020-09-07 20:37:59.110 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 47: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 296.73it/s]
2020-09-07 20:38:09.220 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 47: Training loss 0.181462
2020-09-07 20:38:09.220 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 48: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.82it/s]
2020-09-07 20:38:19.260 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 48: Training loss 0.181292
2020-09-07 20:38:19.260 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 49: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 298.10it/s]
2020-09-07 20:38:29.324 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 49: Training loss 0.181203
2020-09-07 20:38:29.324 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 50: Starting training
100%|██████████| 3000/3000 [00:10<00:00, 297.16it/s]
2020-09-07 20:38:39.420 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 50: Training loss 0.181089
```

![autoencoder-curve.png](/assets/autoencoder-curve.png)

> **NOTE**: We can see that the training loss function decreases rapidly initially and then reaches a stable value after ~25 epochs
{: .prompt-tip }

## Testing

Let's look at how the network does with re-constructing the images. We will pass our test images through the model and compare the input and output images. The code to test the images looks fairly simple. We enable eval mode on the model and pass the image batch tensors through the model. The output will be the re-constructed image tensors.

```python
if use_gpu:
    model = model.cuda()
model.eval()
for data, _ in data_provider.test:
    result = model(data)
    yield data, result
```

Let's look at some results:

![conv-autoencoder-3.png](/assets/conv-autoencoder-3.png)

![conv-autoencoder-4.png](/assets/conv-autoencoder-4.png)

> **NOTE**: We do a lot better with images with simple pixel structures such as T-shirts, dresses, sneakers. We don't do that well with intricate pixel structures such as heals and patterned T-shirts
{: .prompt-info }

## Github Link

You can access the full project on [my Github](https://github.com/kapilsh/ml-projects).
