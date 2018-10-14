# Build a Traffic Sign Recognition classifier
as part of Udacity - Self-Driving Car NanoDegree

## Overview
In this project, Deep learning , specifically convolutional neural networks are used to recognize and classify traffic signs. The model is trained and validated on traffic sign images from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).


## Dependencies
This project requires Python 3.5 and the following Python libraries installed:


* NumPy
* glob
* pickle
* opencv
* TensorFlow
* Matplotlib
* Pandas

## Set up your environment

### 1. Install dependencies

**Create** carnd-term1 environment. Running this command will create a new conda environment that is provisioned with all libraries you need for this project.
```
conda env create -f environment.yml
```
**Verify** that the carnd-term1 environment was created in your environments:
```
conda info --envs
```
### 2. TensorFlow
If you have access to a GPU, you should follow the TensorFlow instructions for [installing TensorFlow with GPU support](https://www.tensorflow.org/get_started/os_setup#optional_install_cuda_gpus_on_linux).  

## Dataset

1. [Download the dataset](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which the images are resized to 32x32x3.
2. Extract the compressed file into `traffic_signs_data` directory to hold `train.p` , `valid.p` and `test.p` datasets. 
3. Clone the project and follow the instruction in the Ipython notebook.







