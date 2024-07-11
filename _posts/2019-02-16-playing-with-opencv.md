---
title: Playing with OpenCV
description: >-
  Face Detection using OpenCV in Python
date: 2019-02-16
categories: [Blog, Tutorial]
tags: [Python, Machine Learning]
pin: false
author: ks
---

I played around with OpenCV in python to experiment with face detection in images. In this post I will cover:
- How to read image files into numpy array
- Detect face in an image
- Mark detected faces
- Experiment on human and dog faces

Firstly, let us download the images. The images can be downloaded from [Dog Images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and [Human Images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). I downloaded the images, unzipped them, and put them in /data/ (my big second hard drive) on my linux box. Depending on where you download images yourself, change the directory in the code snippets below.

## Loading Images

Loading images is very simple with OpenCV and images are loaded as numpy arrays

```python
import cv2
cv2.imread(file_name)
file_name='/data/dog_images/train/124.Poodle/Poodle_07929.jpg'
image = cv2.imread(file_name)
print(image)
```

```
array([[[135, 176, 155],
        [126, 167, 146],
        [107, 151, 128],
        ...,
        [ 72,  92, 109],
        [ 69,  89, 106],
        [ 65,  85, 102]]], dtype=uint8)
```

You can also convert images into different color schemes. For example,

```python
img = cv2.imread(file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

## Face Detection
Now let's start using the tools that are part of OpenCV to detect faces in images. OpenCV ships with the CascadeClassifier, which is an ensemble model used for image processing tasks such as object detection and tracking, primarily facial detection and recognition

In the following snippet, we will initialize the cascade classifier and use it to detect faces. If we detect any faces, we will mark a blue rectangle around the detected edges.

```python
DetectedFace = namedtuple("DetectedFace", ["faces", "image"])

def detect_faces(file_name: str) -> DetectedFace:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return DetectedFace(faces=faces, image=cv_rgb)
```


If we detect any faces, faces field in DetectedFace will have non zero length. Below we define a function to check whether we detected any faces or not:

```python
def face_present(file_path: str) -> bool:
    img = detect_faces(file_path)
    return len(img.faces) > 0
```

Finally, we define the function to plot the marked image.

```python
def plot_detected_faces(img: DetectedFace):
    fig, ax = plt.subplots()
    ax.imshow(img.image)
    plt.show()
```

Let's use the code above to test some sample images. We load up downloaded images into an array and choose a random image to test our face detector. First, we try on a random human image and then on a random dog image.

### Human Images

```python
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

image = detect_faces(human_files[np.random.randint(0, len(human_files))])
plot_detected_faces(image)
```

![face-detection-1.png](/assets/face-detection-1.png)

### Dog Images

```python
image = detect_faces(dog_files[np.random.randint(0, len(dog_files))])
plot_detected_faces(image)
```

![face-detection-2.png](/assets/face-detection-2.png)

Now, let's run the same code on a bunch of different images that we downloaded. I have selected 1000 images from both sets to run the face detector.

```python
def plot_detected_faces_multiple(results: List[DetectedFace],
                                 rows: int = 3, columns: int = 3):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    for r in range(rows):
        for c in range(columns):
            img = results[r * columns + c]
            ax[r][c].imshow(img.image)
    plt.show()


filter_count = 1000

human_images_result = list(map(
    detect_faces,
    human_files[np.random.randint(0, len(human_files), filter_count)]))
dog_images_result = list(map(
    detect_faces,
    dog_files[np.random.randint(0, len(dog_files), filter_count)]))

plot_detected_faces_multiple(human_images_result)
plot_detected_faces_multiple(dog_images_result)
```

![face-detection-4.png](/assets/face-detection-4.png)

As we can see, it does a great job at detecting human faces in all the images, even when there are multiple humans in an image.

...

On the dog images, not so much...

![face-detection-3.png](/assets/face-detection-3.png)

## Final Comments

This was a fun and small project to play around with OpenCV's image processing toolkit. My next goal is to have a better dog image classifier using CNN.
