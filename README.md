# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals/steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dataset]: ./examples/dataset.png "Dataset"
[aug1]: ./examples/aug1.png "Augmented Image 1"
[aug2]: ./examples/aug2.png "Augmented Image 2"
[aug3]: ./examples/aug3.png "Augmented Image 3"
[sign1]: ./german-signs/1.jpg "Traffic Sign 1"
[sign2]: ./german-signs/2.jpg "Traffic Sign 2"
[sign3]: ./german-signs/3.jpg "Traffic Sign 3"
[sign4]: ./german-signs/4.jpg "Traffic Sign 4"
[sign5]: ./german-signs/5.jpg "Traffic Sign 5"
[predictions]: ./examples/predictions.png "Predictions"
[visualization]: ./examples/visualization.png "Visualization"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/istepanov/Udacity-CarND-P2-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

![Dataset][dataset]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

All input images are normalized and grayscaled. Normalization means that every pixel value is converted to be in interval `[-0.5, 0.5]` instead of `[0, 255]`. That makes mean value 0 and small mean squared deviation. Normalized data helps neural network to be trained faster and reduces the chances of getting stuck in local optima. Grayscaling simply reduces the number of input data the neural network needs to learn (1-pixel depth instead of 3), that reduces network size and makes training faster.

To make a larger training set I've generated some additional data by augmenting existing training set (randomly choosing 25% images to augment):

![Augmented Image 1][aug1] ![Augmented Image 2][aug2] ![Augmented Image 3][aug3]

The augmentation includes random scaling, rotating, translating and changing image gamma.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is basically the classic LeNet architecture plus 50% dropout layers between fully-connected layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 grayscale image                       |
| Convolution 1x1       | 1x1 stride, valid padding, outputs 28x28x6    |
| ReLU                  |                                               |
| Max pooling           | 2x2 stride, outputs 14x14x6                   |
| Convolution 1x1       | 1x1 stride, valid padding, outputs 10x10x16   |
| ReLU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected       | Input - 400, output - 120                     |
| ReLU                  |                                               |
| Dropout               | 50%                                           |
| Fully connected       | Input - 120, output - 84                      |
| ReLU                  |                                               |
| Dropout               | 50%                                           |
| Fully connected       | Input - 84, output - 46                       |
| Softmax               |                                               |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Hyperparameters       | Value                                         |
|:---------------------:|:---------------------------------------------:|
| LEARNING_RATE         | 0.001                                         |
| EPOCHS                | 20                                            |
| BATCH SIZE            | 512                                           |
| DROPOUT               | 0.5                                           |

I've chosen Adam Optimizer since it looks shows really good performance, according to [this article](http://sebastianruder.com/optimizing-gradient-descent/).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 82%
* validation set accuracy of 93.9%
* test set accuracy of 92%

Note: as we can see training accuracy is lower than validation accuracy, which is quite unusual. My hypothesis: training set is harder to classify because it has distorted augmented images.

I've chosen LeNet architecture because it works pretty well for small grayscale images. It has much fewer parameters to train (less than 1M) compare to modern "heavy-lifters" like AlexNet (60M) and VGGNet (140M). Combining with modern GPU (I used nVidia GTX 970) the training process took around 20 seconds.

Real-life usage might require higher-res images for better performance, so it might be better to use more modern network architecture.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][sign1]<br />
![Traffic Sign 2][sign2]<br />
![Traffic Sign 3][sign3]<br />
![Traffic Sign 4][sign4]<br />
![Traffic Sign 5][sign5]

They all have different sizes, so they were downscaled to 32x32 pixels before feeding into the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                                         |     Prediction                                |
|:---------------------------------------------:|:---------------------------------------------:|
| No passing for vehicles over 3.5 metric tones | No passing for vehicles over 3.5 metric tones |
| Priority road                                 | Priority road                                 |
| Speed limit (30 km/h)                         | Speed limit (30 km/h)                         |
| Bumpy road                                    | Bumpy Road                                    |
| Stop                                          | Stop                                          |

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![Predictions][predictions]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

These are the first conv layer activations for "No passing for vehicles over 3.5 metric tones" sign:

![Visualization][visualization]

Round shape, white background, and icon in the center - these characteristics are used by the neural network to classify this sign.
