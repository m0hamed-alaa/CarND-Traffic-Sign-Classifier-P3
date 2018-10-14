# **Traffic Sign Recognition** 

This is a writeup for traffic sign recognition project . 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[train]: ./outputs/Distribution_of_traffic_signs_in_the_training_set.png "Training set"
[valid]: ./outputs/Distribution_of_traffic_signs_in_the_validation_set.png "Validation set"
[test]:  ./outputs/Distribution_of_traffic_signs_in_the_test_set.png "Test set" 
[samples]: ./outputs/traffic_signs.png "samples of traffic signs"
[shifting]: ./outputs/shifting.png "shifting"
[rotating]: ./outputs/rotating.png "rotating"
[normalizing]: ./outputs/normalization.png "normalization"

[image1]: ./test_images/30kmh.jpg "Traffic Sign 1"
[image2]: ./test_images/Right_of_way_at_the_next_intersection "Traffic Sign 2"
[image3]: ./test_images/Road_Work.jpg "Traffic Sign 3"
[image4]: ./test_images/Stop_Sign.jpg "Traffic Sign 4"
[image5]: ./test_images/yield.jpg "Traffic Sign 5"

[top5]: ./outputs/top_guesses.png "top guesses"
[bar]: ./outputs/softmax_probabilitis.png "bar charts"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. This github repository includes these files :
* Ipython notebook with code
* HTML exported version of the Ipython notebook
* A writeup report in markdown - you are reading it !
* `signnames.csv` file storing traffic sign names
* A directory `test_images` containing new traffic sign images to test the model on
* A directory `outputs` containing visualizations of different steps in the code
  
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 
I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799 .
* The size of the validation set is 4410
* The size of test set is 12630 . 
* The shape of a traffic sign image is (32, 32, 3) which is RGB color image .
* The number of unique classes/labels in the data set is 43 .   
`signnames.csv` file contains mapping from each class id to its corresponding traffic sign name .  

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is histogram showing how the distribution of classes 
in the training , validation and test set . It's clear that more examples are of some classes than others.   

* ### Training set
![alt text][train]
 ---
* ### Validation set
![alt text][valid]
 ---
* ### Test set
![alt text][test]

Here is a another visualization of the dataset. It plots a sample of each traffic sign type found in it. 

![alt text][samples]

### Design and Test a Model Architecture

#### 1. Describtion of how I preprocessed the image data. 

I'm going to consider What techniques were chosen and why did I choose these techniques .
Pre-processing refers to techniques such as converting to grayscale, normalization, data augmentation , etc. As a first step, I decided to add more data to the dataset by using Data augmentation technique because it increases the diversity of the training set and ths helps the convolutional neural network be robust to variances in rotations and shifts of objects it learns to classify and it reduces overfitting .
The first method I used is generating more images by introducing random shifts/translations to images of the training set. 

 Here is an example of a traffic sign before and after random shifting . the image on the left is the original image while on the right , the shifted one

![alt text][shifting]

The second method is to generate more images by introducing random rotations to images of the training set.

Here is an example of a traffic sign before and after random rotating within `[ -20 , 20 ]` angles . the image on the left is the original one while on the right , the rotated one.

![alt text][rotating]

* Number of training examples before Data augmentation is 34799 .
* Number of training examples after  Data augmentation is 139196 .


As a last step, I normalized the image data because it ensures that each input parameter (pixel in this case) has similar distribution . This helps the convolutional neural network converge faster while training. Normalization is done by subtracting the mean from each pixel then dividing by the standard deviation. This results in centering the data around zero and making it have equal variances in all dimesnions . 

Here is an example of a traffic sign image before and after normalization.

![alt text][normalizing]


#### 2. Describtion of what the model architecture looks like 

The model architecture consists of the feature extration part which is conv layer -> conv layer -> max pooling layer -> conv layer -> conv layer -> max pooling layer -> conv layer -> max pooling layer -> flatten followed by a non-linear classification part consisting of two fully_connected layers and an output layer .  
The following table summarizes the input dimesnions , layers, layer sizes, activations type , connectivity and output .  

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x16    |
| RELU                  |                                               |
| Max pooling 2x2       | 2x2 stride,valid padding, outputs 16x16x16    |
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x32    |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x64    |
| RELU                  |                                               |
| Max pooling 2x2	    | 2x2 stride,valid padding, outputs 8x8x64      | 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x100     |
| RELU                  |            									|
| Max pooling 2x2  		| 2x2 stride,valid padding, outputs 4x4x100     |
| Flatten               | outputs 1600                                  |
| Fully connected       | input 1600, outputs 640                       |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | input 640 , outputs 200                       |
| RELU                  |                                               |
| Dropout               |                                               |
| Output                | input 200 , outputs 43                        |
| Softmax				|                                               |
 


#### 3. Describtion of how I trained my model  The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with batch size equals 128 and I set the number of epochs to 80 with a learning rate of 0.001 and droput keep probability of 0.5 .

#### 4. The approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.
 Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Deep learning in a empirical process. I started with LeNet architecture as it's a well-known architecture. The performance wasn't that high because of the low-resolution and bad lighting conditions of images in the dataset. I decided to increase the capacity of the model by utilizing several convolutional and pooling layer and two fully-connected layers before the output layer ( the details of the architecture are found above ! ). The training error was low and the validation error was high which implies over-fitting so I had to add droput after each dense layer so that the model can generalize well to unseen examples and reduce over-fitting. With Xavier initialization of filter weights and biases and some fine-tunning of learning rate as I reduced it gradually until it reached 0.001 , the model was able to achieve a validation accuracy of 95.2 %. In order to improve the performance furhter , I considered adding more data to the dataset by using the data augmentation technique so the model may be able to recognize the traffic signs correctly regardless of their position or orientation in the images. This helped the model achieve a validation accuracy of 98.41 %.  

The final model results are found in the notebook in the section entitled ` train,validate and test the model` and they are :
* training set accuracy of 0.9994
* validation set accuracy of 0.9841
* test set accuracy of 0.973

 
### Test a Model on New Images

#### 1. Choosing random five German traffic signs found on the web 

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image3] 
![alt text][image3] 
![alt text][image4] 
![alt text][image5]

#### 2. Discussion of the model's predictions on these new traffic signs 

Here are the results of the prediction on these images

| Image			                         |     Prediction	        					| 
|:--------------------------------------:|:--------------------------------------------:| 
| Yield      	     	                 | Yield   			      				     	| 
| Speed limit (30km/h)                   | Speed limit (30km/h)							|
| Stop					                 | Stop											|
| Right-of-way at the next intersection	 | Right-of-way at the next intersection		|     				 			
| Road work                              | Road work                                    |

The model was able to correctly predict 5 out the 5 traffic signs , which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.3% . 

#### 3. Describe how certain the model is when predicting on each of the five new images

This is investigated by looking at the softmax probabilities for each prediction. The top 5 softmax probabilities for each image is provided with the sign type of each probability along with a bar chart in the second visulaization .

The code for making predictions on these images and plotting the outputs is located in the sections `Predict labels of new  images` and `Analyze performance` at the end of the Ipython notebook. As seen in the images below , the model is pretty confident about its predictions of the 5 German traffic sign images 

![alt text][bar]
![alt text][top5]