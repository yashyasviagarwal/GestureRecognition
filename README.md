##
# **Deep Learning Course Project- Gesture Recognition**

- **Maniraj Madishetty – Group Facilitator**
- **Yashyasvi Agarwal**

# Problem Statement

As a data scientist at a home electronics company which manufactures state of the art smart televisions. We want to develop a cool feature in the smart-TV that can recognize five different gestures performed by the user which will help users control the TV without using a remote.

- Thumbs up :  Increase the volume.
- Thumbs down : Decrease the volume.
- Left swipe : 'Jump' backwards 10 seconds.
- Right swipe : 'Jump' forward 10 seconds.
- Stop : Pause the movie.

# Understanding the Dataset

The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a **sequence of 30 frames (images)**. These videos have been recorded by various people performing one of the five gestures in front of a webcam - like what the smart TV will use.

# Objective

Our task is to train different models on the 'train' folder to predict the action performed in each sequence or video and which performs well on the 'val' folder as well. The final test folder for evaluation is withheld - final model's performance will be tested on the 'test' set.

# Two types of architectures suggested for analysing videos using deep learning:

1. **3D Convolutional Neural Networks (Conv3D)**

_3D convolutions_ are a natural extension to the 2D convolutions you are already familiar with. Just like in 2D conv, you move the filter in two directions (_x_ and _y_), in 3D conv, you move the filter in three directions (_x_, _y_ and _z_). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is _100 x 100 x 3_, for example, the video becomes a 4D tensor of shape _100 x 100 x 3 x 30_ which can be written as _(100 x 100 x 30) x 3_ where _3_ is the number of channels. Hence, deriving the analogy from 2D convolutions where a 2D kernel/filter (a square filter) is represented as _(f x f) x c_ where _f_ is filter size and _c_ is the number of channels, a 3D kernel/filter (a _'cubic'_ filter) is represented as _(f x f x f) x c_ (here _c = 3_ since the input images have three channels). This cubic filter will now _'3D-convolve'_ on each of the three channels of the _(100 x 100 x 30)_ tensor

.

1. **CNN + RNN architecture **

The _conv2D_ network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular softmax (for a classification problem such as this one).

# Data Generator

This is one of the most important part of the code. In the generator, we are going to pre-process the images as we have images of 2 different dimensions (_360 x 360_ and _120 x 160_) as well as create a batch of video frames. The generator should be able to take a batch of videos as input without any error. Steps like cropping, resizing and normalization should be performed successfully.

# Data Pre-processing

- _ **Resizing** _ **and** _ **cropping** _ **of the images.** This was mainly done to ensure that the NN only recognizes the gestures effectively rather than focusing on the other background noise present in the image.
- _ **Normalization** _ **of the images.** Normalizing the RGB values of an image can at times be a simple and effective way to get rid of distortions caused by lights and shadows in an image.
- At the later stages for improving the model's accuracy, we have also made use of _ **data augmentation** _, where we have _ **slightly rotated** _ the pre-processed images of the gestures in order to bring in more data for the model to train on and to make it more generalizable in nature as sometimes the positioning of the hand won't necessarily be within the camera frame always.

![](RackMultipart20221218-1-507p9d_html_60475b7632d233a3.png) ![](RackMultipart20221218-1-507p9d_html_46728caecc39496b.png)

# NN Architecture development and training

- Experimented with different model configurations and hyper-parameters and various iterations and combinations of batch sizes, image dimensions, filter sizes, padding and stride length were experimented with. We also played around with different learning rates and _ReduceLROnPlateau_ was used to decrease the learning rate if the monitored metrics (_val\_loss_) remains unchanged in between epochs.
- We experimented with _SGD()_ and _Adam()_ optimizers but went forward with _Adam()_ as it lead to improvement in model's accuracy by rectifying high variance in the model's parameters.
- We also made use of _Batch Normalization_, _pooling_ and _dropout__layers_ when our model started to overfit, this could be easily witnessed when our model started giving poor validation accuracy in spite of having good training accuracy.

# Observations

- It was observed that as the Number of trainable parameters increase, the model takes much more time for training.
- **Batch size**  **∝**  **GPU memory / available compute.** A large batch size can throw _GPU Out of memory error,_ and thus here we had to play around with the batch size till we were able to arrive at an optimal value of the batch size which our GPU could support
- Increasing the batch size greatly reduces the training time but this also has a negative impact on the model accuracy. This made us realise that there is always a trade-off here on basis of priority -\> If we want our model to be ready in a shorter time span, choose larger batch size else you should choose lower batch size if you want your model to be more accurate.
- _Data Augmentation_ greatly helped in overcoming the problem of overfitting which our initial version of model was facing.
- _CNN+LSTM_ based model had better performance than _Conv3D._ As per our understanding, this is something which depends on the kind of data we used, the architecture we developed and the hyper-parameters we chose.
- _Transfer learning_ **boosted** the overall accuracy of the model. We made use of the [_MobileNet_](https://arxiv.org/abs/1704.04861) Architecture due to its lightweight design and high speed performance coupled with low maintenance as compared to other well-known architectures like VGG16, AlexNet, GoogleNet et

Model Experimentation Results:

| **Experiment Number** | **Model** | **Result** | **Decision + Explanation** |
| --- | --- | --- | --- |
| **1** | **Conv3D** | **OOM Error** | **Reduce the batch size and crop the images correctly, try to overfit on less amount of data** |
| **2** | **Conv3D** | **Training Accuracy: 0.99**** Validation Accuracy: 0.81 **|** Overfitting **** Let's add some Dropout Layers** |
| **3** | **Conv3D** | **Training Accuracy: 0.72**** Validation Accuracy: 0.79 **|** Adding dropouts has caused validation accuracy to increase more than training accuracy. Let us move to CNN+LSTM** |
| **4** | **CNN+LSTM** | **Training Accuracy: 0.83**  **Validation Accuracy: 0.78** | **Although the accuracy has improved it is still low, we might need to induce transfer learning** |
| **Final Model** | **Transfer Learning + GRU** | **Training Accuracy: 94**** Validation Accuracy: 97 **|** Transfer learning with learnt weights worked well and we are getting very high accuracy.** |

After going through all the models we finally decide to use the model with transfer learning as it gives us the most accuracy on both Train and Validation set. This model is performing very accurately and can be used as the final model in production.