---
title: Dog Breed Classification
description: >-
  Train a Convolutional Neural Network to Detect Dog Breeds
date: 2019-03-22
categories: [Blog, Tutorial]
tags: [Python, Machine Learning, Deep Learning, CNN]
pin: true
author: ks
---

In continuation to the [previous post](../dog-detection/), where I played around with PyTorch to detect dog images, in this post, I will train a Convolutional Neural Network to classify breeds of dogs using PyTorch. The images for the project can be downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

## Loading Data

Our initial step before we start training is to define our data loaders. For this, we will be using [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#module-torch.utils.data). We need to provide the path for the training, validation and testing images. As part of the training, we will need to apply some transformations to the data such as resizing, random flips, rotations, normalization, etc. We can apply the transformation as part of the data loader so that transformations are already handled when images are fed into the network. Images can be trained in batches, so we will also provide the batch size for when they are fed to the model.

```python
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_loader = DataLoader(
    datasets.ImageFolder(
        "/data/dog_images/train",
        transform=transform_train),
    shuffle=True,
    batch_size=64,
    num_workers=6)
```

I defined a class that provides access to train, test and validation datasets as below:

### DataProvider

```python
class DataProvider:
    def __init__(self, root_dir: str, **kwargs):
        self._root_dir = root_dir
        self._train_subfolder = kwargs.pop("train_subfolder", "train")
        self._test_subfolder = kwargs.pop("test_subfolder", "test")
        self._validation_subfolder = kwargs.pop("validation_subfolder", "valid")
        self._batch_size = kwargs.pop("batch_size", 64)
        self._num_workers = kwargs.pop("num_workers", 0)

        logger.info(f"ROOT_DIR: {self._root_dir}")
        logger.info(f"BATCH_SIZE: {self._batch_size}")
        logger.info(f"NUM WORKERS: {self._num_workers}")

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        transform_others = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self._train_loader = DataLoader(
            datasets.ImageFolder(
                os.path.join(root_dir, self._train_subfolder),
                transform=transform_train),
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_workers)

        self._validation_loader = DataLoader(
            datasets.ImageFolder(
                os.path.join(root_dir, self._validation_subfolder),
                transform=transform_others),
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_workers)

        self._test_loader = DataLoader(
            datasets.ImageFolder(os.path.join(root_dir, self._test_subfolder),
                                 transform=transform_others),
            shuffle=False,
            batch_size=self._batch_size,
            num_workers=self._num_workers)

    @property
    def train(self) -> DataLoader:
        return self._train_loader

    @property
    def test(self) -> DataLoader:
        return self._test_loader

    @property
    def validation(self) -> DataLoader:
        return self._validation_loader
```

## Architecture

This is a practical post, so I won't get into too much detail on how I came up with the architecture. However, the general idea is to apply convolutional filters to the image to capture spatial features from the image and then use pooling to enhance the features after each layer. I will be using 5 convolutional layers, with each convolutional layer followed by a max-pooling layer. The convolutional layers are followed by 2 fully-connected layers with ReLU activation and dropout in the middle.

I defined the architecture using a `nn.Module` implementation.

### Neural Net Architecture

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, (3, 3))
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, (3, 3))
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(5 * 5 * 256, 400)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(400, 133)

    def forward(self, x):
        x = self.pool1(functional.relu(self.conv1(x)))
        x = self.pool2(functional.relu(self.conv2(x)))
        x = self.pool3(functional.relu(self.conv3(x)))
        x = self.pool4(functional.relu(self.conv4(x)))
        x = self.pool5(functional.relu(self.conv5(x)))

        x = x.view(-1, 5 * 5 * 256)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

 ![dog-breed-classification-4.png](/assets/dog-breed-classification-4.png)

## Model Training and Validation
Now that we have our data loader and model defined, the next step is to create the training and validation routine. I will be doing the training and validation in a single step, checking the validation loss at each epoch to check whether we have reduced validation loss.

During the training phase, you will have to activate training mode in PyTorch by calling `model.train()` on the model instance. After that, we need to follow the following steps:
- Zero out optimizer gradients
- Apply forward pass or `model(input)`
- Calculate the loss function
- Apply the backward pass or `loss.backward()`
- Take new optimization step

I am using Cross Entropy Loss function and Adam optimizer for training the ConvNet. The steps look like below in code:

```python
neural_net.train()
for batch_index, (data, target) in enumerate(
        self._data_provider.train):
    logger.debug(f"[TRAIN] Processing Batch: {batch_index}")
    if self._use_gpu:
        data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()
    output = neural_net(data)
    loss = self._criterion(output, target)
    loss.backward()
    optimizer.step()
    train_loss = train_loss + (
            (loss.item() - train_loss) / (batch_index + 1))
```

As we are training batches, each iteration will pass multiple images to the training loop. I am using a batch size of 64 hence the input set of images would look like:

![dog-breed-classification-1.png](/assets/dog-breed-classification-1.png)

Validation step looks similar but simpler, where we dont need to calculate any gradients and just calculate loss function from the forward pass.

```python
neural_net.eval()
for batch_index, (data, target) in enumerate(
        self._data_provider.validation):
    logger.debug(f"[VALIDATE] Processing Batch: {batch_index}")
    if self._use_gpu:
        data, target = data.cuda(), target.cuda()

    with torch.no_grad():
        output = neural_net(data)
    loss = self._criterion(output, target)
    validation_loss = validation_loss + (
            (loss.item() - validation_loss) / (batch_index + 1))
```

One thing to note is that if you are training on a GPU, you need to move your data and model to the GPU by calling .cuda() assuming training on NVidia GPUs.

I save the model, everytime the validation loss decreases. So, we will automatically have the model with the minimum validation loss by the end of the training loop.

### Training/Validation Logs

```
2019-03-02 21:57:47.630 | INFO     | __main__:__init__:28 - ROOT_DIR: /data/dog_images/
2019-03-02 21:57:47.630 | INFO     | __main__:__init__:29 - BATCH_SIZE: 64
2019-03-02 21:57:47.630 | INFO     | __main__:__init__:30 - NUM WORKERS: 0
2019-03-02 21:57:47.673 | INFO     | __main__:__init__:138 - CUDA is enabled - using GPU
2019-03-02 21:57:49.294 | INFO     | __main__:train:147 - Model Architecture: 
NeuralNet(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (pool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=6400, out_features=400, bias=True)
  (dropout): Dropout(p=0.3, inplace=False)
  (fc2): Linear(in_features=400, out_features=133, bias=True)
)
2019-03-02 21:57:49.294 | INFO     | __main__:_train_epoch:175 - [Epoch 0] Starting training phase
2019-03-02 21:58:52.011 | INFO     | __main__:_train_epoch:189 - [Epoch 0] Starting eval phase
2019-03-02 21:58:58.073 | INFO     | __main__:train:159 - Validation Loss Decreased: inf => 4.704965. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 21:58:58.096 | INFO     | __main__:_train_epoch:175 - [Epoch 1] Starting training phase
2019-03-02 22:00:01.268 | INFO     | __main__:_train_epoch:189 - [Epoch 1] Starting eval phase
2019-03-02 22:00:07.681 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.704965 => 4.440237. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:00:07.700 | INFO     | __main__:_train_epoch:175 - [Epoch 2] Starting training phase
2019-03-02 22:01:10.125 | INFO     | __main__:_train_epoch:189 - [Epoch 2] Starting eval phase
2019-03-02 22:01:16.491 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.440237 => 4.318264. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:01:16.514 | INFO     | __main__:_train_epoch:175 - [Epoch 3] Starting training phase
2019-03-02 22:02:19.505 | INFO     | __main__:_train_epoch:189 - [Epoch 3] Starting eval phase
2019-03-02 22:02:25.860 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.318264 => 4.110924. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:02:25.886 | INFO     | __main__:_train_epoch:175 - [Epoch 4] Starting training phase
2019-03-02 22:03:27.334 | INFO     | __main__:_train_epoch:189 - [Epoch 4] Starting eval phase
2019-03-02 22:03:33.551 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.110924 => 4.011838. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:03:33.574 | INFO     | __main__:_train_epoch:175 - [Epoch 5] Starting training phase
2019-03-02 22:04:37.324 | INFO     | __main__:_train_epoch:189 - [Epoch 5] Starting eval phase
2019-03-02 22:04:43.689 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.011838 => 3.990763. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:04:43.718 | INFO     | __main__:_train_epoch:175 - [Epoch 6] Starting training phase
2019-03-02 22:05:46.730 | INFO     | __main__:_train_epoch:189 - [Epoch 6] Starting eval phase
2019-03-02 22:05:53.396 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.990763 => 3.849251. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:05:53.421 | INFO     | __main__:_train_epoch:175 - [Epoch 7] Starting training phase
2019-03-02 22:06:58.002 | INFO     | __main__:_train_epoch:189 - [Epoch 7] Starting eval phase
2019-03-02 22:07:04.432 | INFO     | __main__:_train_epoch:175 - [Epoch 8] Starting training phase
2019-03-02 22:08:07.631 | INFO     | __main__:_train_epoch:189 - [Epoch 8] Starting eval phase
2019-03-02 22:08:14.045 | INFO     | __main__:_train_epoch:175 - [Epoch 9] Starting training phase
2019-03-02 22:09:16.695 | INFO     | __main__:_train_epoch:189 - [Epoch 9] Starting eval phase
2019-03-02 22:09:23.421 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.849251 => 3.717872. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:09:23.445 | INFO     | __main__:_train_epoch:175 - [Epoch 10] Starting training phase
2019-03-02 22:10:27.581 | INFO     | __main__:_train_epoch:189 - [Epoch 10] Starting eval phase
2019-03-02 22:10:34.121 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.717872 => 3.588202. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:10:34.144 | INFO     | __main__:_train_epoch:175 - [Epoch 11] Starting training phase
2019-03-02 22:11:38.351 | INFO     | __main__:_train_epoch:189 - [Epoch 11] Starting eval phase
2019-03-02 22:11:44.652 | INFO     | __main__:_train_epoch:175 - [Epoch 12] Starting training phase
2019-03-02 22:12:47.946 | INFO     | __main__:_train_epoch:189 - [Epoch 12] Starting eval phase
2019-03-02 22:12:54.915 | INFO     | __main__:_train_epoch:175 - [Epoch 13] Starting training phase
2019-03-02 22:13:58.543 | INFO     | __main__:_train_epoch:189 - [Epoch 13] Starting eval phase
2019-03-02 22:14:04.912 | INFO     | __main__:_train_epoch:175 - [Epoch 14] Starting training phase
2019-03-02 22:15:07.638 | INFO     | __main__:_train_epoch:189 - [Epoch 14] Starting eval phase
2019-03-02 22:15:14.058 | INFO     | __main__:_train_epoch:175 - [Epoch 15] Starting training phase
2019-03-02 22:16:17.191 | INFO     | __main__:_train_epoch:189 - [Epoch 15] Starting eval phase
2019-03-02 22:16:23.634 | INFO     | __main__:_train_epoch:175 - [Epoch 16] Starting training phase
2019-03-02 22:17:26.982 | INFO     | __main__:_train_epoch:189 - [Epoch 16] Starting eval phase
2019-03-02 22:17:33.304 | INFO     | __main__:_train_epoch:175 - [Epoch 17] Starting training phase
2019-03-02 22:18:36.207 | INFO     | __main__:_train_epoch:189 - [Epoch 17] Starting eval phase
2019-03-02 22:18:42.790 | INFO     | __main__:_train_epoch:175 - [Epoch 18] Starting training phase
2019-03-02 22:19:46.360 | INFO     | __main__:_train_epoch:189 - [Epoch 18] Starting eval phase
2019-03-02 22:19:53.077 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.588202 => 3.558330. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model
2019-03-02 22:19:53.106 | INFO     | __main__:_train_epoch:175 - [Epoch 19] Starting training phase
2019-03-02 22:20:57.718 | INFO     | __main__:_train_epoch:189 - [Epoch 19] Starting eval phase
2019-03-02 22:21:04.343 | INFO     | __main__:train:248 - Training Results: TrainedModel(train_losses=[4.828373968033565, 4.561895974477133, 4.3049539429800845, 4.10613343602135, 3.9616453170776356, 3.837490134012132, 3.729485934121267, 3.6096336637224464, 3.4845925603594092, 3.390888084684101, 3.2799783706665036, 3.20562988917033, 3.072563396181379, 2.9623924732208247, 2.870406493686495, 2.7523970808301663, 2.665678980236962, 2.535139397212437, 2.430639664332072, 2.333072783833458], validation_losses=[4.704964978354318, 4.440237283706664, 4.318263803209577, 4.110924073628017, 4.011837703841073, 3.990762727601187, 3.8492509978158136, 3.8797887223107472, 3.911121691976275, 3.717871563775199, 3.5882019826344083, 3.6028132949556624, 3.6062802246638705, 3.741273845945086, 3.6166011095047, 3.5896864277975893, 3.968828797340393, 3.668894120625087, 3.558329514094762, 3.6221354859215875], optimal_validation_loss=3.558329514094762)
Process finished with exit code 0
```

![dog-breed-classify-lc.png](/assets/dog-breed-classify-lc.png)

> The validation loss stops improving much after epoch 10. Training loss keeps decreasing, as expected
{: .prompt-tip }

## Testing

During the testing phase, I load the model that was saved earlier during the training phase to run it over test images. We check the final activation of each image and apply the label based on the category with max activation.

```python
model.eval()
for batch_idx, (data, target) in enumerate(self._data_provider.test):
    if self._use_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = self._criterion(output, target)
    test_loss = test_loss + (
            (loss.data.item() - test_loss) / (batch_idx + 1))
    predicted = output.max(1).indices
    predicted_labels = np.append(predicted_labels, predicted.numpy())
    target_labels = np.append(target_labels, target.numpy())
```

```
2019-03-03 21:54:56.620 | INFO     | breed_classifier:__init__:27 - ROOT_DIR: /data/dog_images/
2019-03-03 21:54:56.620 | INFO     | breed_classifier:__init__:28 - BATCH_SIZE: 64
2019-03-03 21:54:56.620 | INFO     | breed_classifier:__init__:29 - NUM WORKERS: 0
2019-03-03 21:54:56.663 | INFO     | breed_classifier:__init__:137 - CUDA is enabled - using GPU
2019-03-03 21:55:04.558 | INFO     | __main__:test:39 - Test Results: TestResult(test_loss=3.607608267239162, correct_labels=161, total_labels=836)
```

Let's look at some of the prediction results in the images below. <span style="color:red">Red</span> and <span style="color:green">Green</span> boxes indicate incorrect and correct predictions, respectively.

![dog-breed-classification-3.png](/assets/dog-breed-classification-3.png)

![impressed.gif](/assets/impressed.gif){: width="200" }

> **Result**: The final test accuracy is around 19%, which is... well not that good but... not that bad either considering we didn't use a very deep CNN. The final test loss value is close to our final validation loss value
{: .prompt-info }

> **Spoiler Alert:** We can do much better than this using Transfer Learning on a pre-trained model like VGG-16, that we used to detect dog images in the precious post
{: .prompt-warning }

### Model Training

The final Model class looks like below:

```python
TrainedModel = namedtuple(
    "TrainedModel",
    ["train_losses", "validation_losses", "optimal_validation_loss"])

TestResult = namedtuple(
    "TestResult", ["test_loss", "correct_labels", "total_labels"])


class Model:
    def __init__(self, data_provider: DataProvider,
                 save_path: str, **kwargs):
        self._data_provider = data_provider
        self._save_path = save_path
        self._criterion = nn.CrossEntropyLoss()
        self._use_gpu = kwargs.pop("use_gpu",
                                   False) and torch.cuda.is_available()
        if self._use_gpu:
            logger.info("CUDA is enabled - using GPU")
        else:
            logger.info("GPU Disabled: Using CPU")

    def train(self, n_epochs) -> TrainedModel:

        neural_net = NeuralNet()
        if self._use_gpu:
            neural_net = neural_net.cuda()
        logger.info(f"Model Architecture: \n{neural_net}")
        optimizer = optim.Adam(neural_net.parameters())
        validation_losses = []
        train_losses = []
        min_validation_loss = np.Inf

        for epoch in range(n_epochs):
            train_loss, validation_loss = self._train_epoch(epoch, neural_net,
                                                            optimizer)
            validation_losses.append(validation_loss)
            train_losses.append(train_loss)
            if min_validation_loss > validation_loss:
                logger.info(
                    "Validation Loss Decreased: {:.6f} => {:.6f}. "
                    "Saving Model to {}".format(
                        min_validation_loss, validation_loss, self._save_path))
                min_validation_loss = validation_loss
                torch.save(neural_net.state_dict(), self._save_path)

        optimal_model = NeuralNet()
        optimal_model.load_state_dict(torch.load(self._save_path))
        return TrainedModel(train_losses=train_losses,
                            validation_losses=validation_losses,
                            optimal_validation_loss=min_validation_loss)

    def _train_epoch(self, epoch: int, neural_net: nn.Module,
                     optimizer: optim.Optimizer):
        train_loss = 0
        logger.info(f"[Epoch {epoch}] Starting training phase")
        neural_net.train()
        for batch_index, (data, target) in enumerate(
                self._data_provider.train):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = neural_net(data)
            loss = self._criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + (
                    (loss.item() - train_loss) / (batch_index + 1))

        logger.info(f"[Epoch {epoch}] Starting eval phase")

        validation_loss = 0
        neural_net.eval()
        for batch_index, (data, target) in enumerate(
                self._data_provider.validation):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = neural_net(data)
            loss = self._criterion(output, target)
            validation_loss = validation_loss + (
                    (loss.item() - validation_loss) / (batch_index + 1))

        return train_loss, validation_loss

    def test(self) -> TestResult:
        model = NeuralNet()
        model.load_state_dict(torch.load(self._save_path))
        if self._use_gpu:
            model = model.cuda()
        test_loss = 0
        predicted_labels = np.array([])
        target_labels = np.array([])

        model.eval()
        for batch_idx, (data, target) in enumerate(self._data_provider.test):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = self._criterion(output, target)
            test_loss = test_loss + (
                    (loss.data.item() - test_loss) / (batch_idx + 1))
            predicted = output.max(1).indices
            predicted_labels = np.append(predicted_labels, predicted.numpy())
            target_labels = np.append(target_labels, target.numpy())

        return TestResult(test_loss=test_loss,
                          correct_labels=sum(np.equal(target_labels,
                                                      predicted_labels)),
                          total_labels=len(target_labels))
```

## Final Thoughts

In this post, I trained a convolutional neural net from scratch to classify dog breeds. In a follow up post, my plan is to use transfer learning to significantly improve the accuracy of the model.


