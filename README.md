# NeuroDigits

# Handwritten Digit Recognition Using Neural Networks

## Overview
This project implements a **Neural Network-based Handwritten Digit Recognition Model** using **TensorFlow** and **Keras**. The model is designed to classify digits from 0 to 9, making use of the **ReLU activation function** for hidden layers and the **Softmax function** for classification. The dataset used in this project is a subset of the **MNIST handwritten digit dataset**, which contains grayscale images of handwritten digits.

---

## Introduction
Handwritten digit recognition is an essential application of deep learning, commonly used in postal services, banking, and automated document processing. This project builds a neural network capable of classifying handwritten digits with high accuracy. The model consists of multiple layers, utilizing **ReLU** activations and the **Softmax function** to output class probabilities.

## Dataset
The dataset consists of:
- **5000 training examples**, each representing a **20x20 grayscale image** of a digit.
- Each image is **flattened into a 400-dimensional vector**.
- The labels (0-9) are stored in a corresponding vector **y**.

The dataset is a subset of the famous MNIST dataset, widely used in machine learning research.

## Model Architecture
The neural network consists of **three layers**:

1. **Input Layer**: 400 neurons (corresponding to the 20x20 pixel input image).
2. **Hidden Layer 1**: 25 neurons with **ReLU activation**.
3. **Hidden Layer 2**: 15 neurons with **ReLU activation**.
4. **Output Layer**: 10 neurons (one for each digit) with **linear activation** (Softmax is applied separately for numerical stability).

### Parameter Dimensions:
| Layer | Weight Matrix Shape | Bias Shape |
|--------|--------------------|------------|
| Hidden Layer 1 | (400, 25) | (25,) |
| Hidden Layer 2 | (25, 15) | (15,) |
| Output Layer | (15, 10) | (10,) |

## Implementation
The project is implemented using **TensorFlow** and **Keras**. The main steps include:

1. **Defining the model** using Keras' `Sequential` API.
2. **Using ReLU activations** for hidden layers and applying Softmax for classification.
3. **Compiling the model** with `SparseCategoricalCrossentropy` as the loss function.
4. **Training the model** using `Adam` optimizer.
5. **Evaluating model performance** and visualizing predictions.

## Training
The model is trained for **40 epochs** with **5000 training samples**. The loss function used is **Sparse Categorical Crossentropy**, which is suited for multi-class classification tasks. 

### Optimizer:
- **Adam Optimizer** (`learning_rate = 0.001`)

### Loss Function:
- `SparseCategoricalCrossentropy(from_logits=True)`

## Evaluation
After training, the model's accuracy is evaluated by predicting digits on unseen samples. The final performance is visualized by comparing actual vs. predicted labels using `argmax` on the Softmax output.

## Results
The model achieves a high accuracy, correctly classifying **99.7% of the test images**, with only **15 misclassifications out of 5000 images**. Increasing the number of training epochs further enhances accuracy.

## Next Steps

- Increase training epochs to further improve accuracy.
- Implement Convolutional Neural Networks (CNNs) for better feature extraction.
- Train on a larger dataset such as full MNIST.
- Experiment with transfer learning using pre-trained CNN models.
- Expand the project to also include letters, so that eventually, it can start scanning on what is in a page or paragraph, rather than a single digit or alphabet.
