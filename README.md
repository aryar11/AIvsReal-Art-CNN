# AI art vs Human Art CNN

In this repository, I have my Jupyter Notebook and my data for my Convolution Neural Network. 

## Dataset
This dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?resource=download) with 540 AI generated images with 436 human made images.

## Jupyter Notebook Outline
1. Crop all Images to the Same Size.
   For all the images in the dataset, I decided to crop or pad them to 720x720.

3. Organize the Images into Train and Test Folders.
   Then, I organized the data into train and test folders.

5. Create Train and Validation Sets. I used a 70/30 split for training and validation data

6. CNN Architecture Summary:
Uses a 3-channel input (for RGB images), a convolutional layer with a kernel size of 3x3, stride of 1, and padding to maintain dimensionality, followed by ReLU activation and max pooling for down-sampling. Dropout of 0.2 is applied after pooling to reduce overfitting.The network dynamically calculates the size needed to flatten the output from the convolutional layers into a 1D tensor for the fully connected layers. This is achieved through a dummy input to simulate the forward pass without training. Following the convolutional layers, the network includes several fully connected (dense) layers with ReLU activations and dropout. The architecture transitions from a flattened size to 128 units, then to 64 units, and finally to the output layer of 2 units. 

7. Training the model: 
Using PyTorch and Tensorflow, the model is trained over 150 epochs with Cross Entropy Loss function and Adam opimizer.

8. Testing. After training and validating the CNN, I test its performance against several metrics: Accuracy, Precision, Recall, F1 Score, and its Confusion Matrix
