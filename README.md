# Feed-Forward Neural Network for Digit Classification

A small feed-forward neural network built with TensorFlow/Keras that classifies images of handwritten digits.

## Dataset
Scikit-learn's `load_digits` dataset (1797 images, 8x8 pixels, digits 0 to 9). This is used instead of full MNIST so the notebook runs offline with no download.

## What the notebook does
- Loads the digit images and displays a few examples
- Normalizes pixel values by dividing by 16 (the max value for this dataset) so inputs fall between 0 and 1
- Builds a Keras Sequential model: `Flatten(input_shape=(8,8))`, `Dense(128, relu)`, `Dense(10, softmax)`
- Compiles with the Adam optimizer and sparse categorical crossentropy loss
- Trains for 30 epochs with a validation split
- Evaluates on the held-out test set
- Plots training versus validation accuracy and loss curves
- Shows sample predictions with predicted versus true labels

## Key ideas
- Flatten turns each 8x8 image into a 64-value vector so the dense layers can process it.
- ReLU in the hidden layer lets the network learn non-linear patterns, without it, stacking layers would just collapse into one linear function.
- Softmax in the output layer turns raw scores into a probability distribution over the 10 digit classes.
- Normalizing pixel values to 0-1 helps training stay stable and converge reliably.
- Comparing training and validation curves is how you check for overfitting: if training accuracy keeps rising while validation accuracy flattens or drops, the model is starting to memorize training data instead of generalizing.

## Note on using full MNIST
This notebook uses scikit-learn's digits dataset so it runs offline. To use full MNIST instead, replace the data loading step with `tf.keras.datasets.mnist.load_data()`, change the input shape to `(28, 28)`, and divide pixel values by 255 instead of 16.

## Tools
Python, TensorFlow/Keras, matplotlib

## How to run
```
pip install -r requirements.txt
jupyter notebook neural_network.ipynb
```
Run the cells top to bottom. Training takes under a minute on CPU.
