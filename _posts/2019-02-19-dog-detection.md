---
title: Dog Detection
description: >-
  Dog Detection using pre-trained VGG-16
date: 2019-02-19
categories: [Blog, Tutorial]
tags: [Python, Machine Learning, Deep Learning, CNN]
pin: false
author: ks
---

In a [previous post](../playing-with-opencv/), I played around with OpenCV to detect human faces on images. In this post, I will do something similar with dog images. All the information related to files are in the previous post. The images can be downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

The goal of this post is to use [PyTorch](https://pytorch.org/) pretrained deep-learning models to detect dogs in images. In this post, I will be using pre-trained [VGG-16](https://arxiv.org/abs/1409.1556) model on ImageNet. Same exercise can be done with other models trained on ImageNet such as Inception-v3, ResNet-50, etc. You can find all the PyTorch pre-trained models [here](https://pytorch.org/vision/stable/models.html).

## Setup

Let's start with loading the pre-trained model. If the model does not exist in your cache, PyTorch will download the model but you will only need to do that once unless you delete your model cache. Loading a pre-trained model is a piece of cake - just load it from the module torchvision.models as below:

```python
import torch
import torchvision.models as models

pretrained_vgg16 = models.vgg16(pretrained=True)
print(pretrained_vgg16) # should print all the layers from the VGG-16 model
```

```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

## Implementation

Now, we will look at the implementation of the testing/prediction of images. To read the images, we will continue to use OpenCV, as we used in the previous post. Since OpenCV loads images in BGR format, we will also need to transform the images to RGB format, which PyTorch expects. Below is a function to load the images:

```python
def read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```

To feed the images through the model, we would need to apply a few transformations to the image, which was loaded as a numpy array:

1. [ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor): We would need to transform the image into a PyTorch Tensor, which changes the dimension so the numpy array to match what PyTorch expects i.e. […, H, W] shape.
2. [Resize](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize): We will resize the image to 256x256 image.
3. [CenterCrop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop): We will grab the cropped center of the image of the given size
4. [Normalize](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize): Finally, we will normalize the images using given mean and stds

We can compose all these transformations as below:

```python
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

After transforming the image, we just need to pass the image through the model to get the loss function. Our final predicted category will be the index with max loss. ImageNet categories 151-268 correspond to dogs so all we need to do is get those in predicted indices.

```python
with torch.no_grad(): # No backpropagation required since we are not training
    output = self._model(image)
    predicted = output.argmax()
```

Finally, I put everything into easy-to-use class, with cuda support as well to test images on GPU.

```python
class DogDetector:
    IMAGENET_MIN_INDEX_DOG = 151
    IMAGENET_MAX_INDEX_DOG = 268

    def __init__(self, use_gpu: bool = False):
        self._model = models.vgg16(pretrained=True)
        self._use_cuda = torch.cuda.is_available() and use_gpu
        if self._use_cuda:
            logger.info("CUDA is enabled - using GPU")
            self._model = self._model.cuda()

    @staticmethod
    def _read_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def predict(self, image_path: str) -> int:
        image = self._read_image(image_path)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = preprocess(image).unsqueeze_(0)

        if self._use_cuda:
            image = image.cuda()

        with torch.no_grad():
            output = self._model(image)
            predicted = output.argmax()

        return predicted

    @timeit
    def detect(self, image_path: str) -> bool:
        predicted_index = self.predict(image_path)
        logger.info(f"Predicted Index: {predicted_index}")
        return (self.IMAGENET_MIN_INDEX_DOG <= predicted_index <=
                self.IMAGENET_MAX_INDEX_DOG)
```

Let's test our DogDetector class on all the dog and human images:

```python
dog_files = np.array(glob("/data/dog_images/*/*/*"))
dog_detector = DogDetector(use_gpu=False)
chosen_size = 100
detected = sum(dog_detector.detect(f) for f in dog_files[:chosen_size])
logger.info(f"Dogs detected in {detected} / {chosen_size} = "
            f"{detected * 100 / chosen_size}% images")

# INFO     | __main__:<module>:6 - Dogs detected in 100 / 100 = 100.0% images
```

```python
human_files = np.array(glob("/data/lfw/*/*"))
detected = sum(dog_detector.detect(f) for f in human_files[:chosen_size])
logger.info(f"Dogs detected in {detected} / {chosen_size} = "
            f"{detected * 100 / chosen_size}% images")
# INFO     | __main__:<module>:9 - Dogs detected in 0 / 100 = 0.0% images
```

![woohoo.gif](/assets/woohoo.gif){: width="300" }

## Summary

In this post, I used a pretrained VGG-16 model to test dog images to indicate whether images have dogs or not. In a follow up project, I plan to extend this to use transfer learning to train a deep learning model for predicting dog breeds.


