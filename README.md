### Neural Network from Scratch for Classification

This project implements a simple, feedforward neural network from scratch using NumPy to solve binary classification problems. It demonstrates the core principles of neural networks, including forward propagation, backpropagation, and gradient descent. The model is tested on two classic synthetic datasets: **"moons"** and **"blobs"** to showcase its ability to handle both linearly separable and non-linearly separable data.

-----

### 🚀 Getting Started

To run the project, you need to execute the provided Jupyter Notebook. The notebook contains all the necessary code to generate the datasets, train the neural networks, and visualize the results.

#### Prerequisites

The project relies on standard Python libraries. You can install the required packages using pip:

```bash
pip install numpy scikit-learn matplotlib pandas
```

#### Usage

1.  Open the Jupyter Notebook file `nn.ipynb`.
2.  Run all the cells in the notebook sequentially.
3.  The code will generate two datasets, train a separate neural network for each, and then display the training loss, accuracy, and predicted classifications.

-----

### 📂 Project Structure

  - `nn.ipynb`: The main Jupyter Notebook containing the entire implementation, including data generation, model definition, training, and evaluation.
  - The notebook includes the following sections:
      - **Data Generation**: Code to create the `make_moons` and `make_blobs` datasets.
      - **Activation Functions**: Implementations of ReLU and Sigmoid, along with their derivatives.
      - **Neural Network Class**: A custom class that defines the structure and functionality of the neural network.
      - **Training and Prediction**: Code for training the model on the generated data and making predictions on test data.
      - **Visualization**: Plots showing the original test data and the model's predictions.

-----

### ⚙️ Implementation Details

The `NeuralNetwork` class is designed with a modular approach, allowing for custom layer sizes and activation functions.

  - **Layer Sizes**: The model is configured with an input layer, one hidden layer, and an output layer (`[2, 4, 1]`).
  - **Activation Functions**:
      - **Sigmoid**: Used for the output layer to produce a probability-like output for binary classification.
      - **ReLU**: Used for the hidden layer.
  - **Loss Function**: Mean Squared Error (MSE) is used to calculate the difference between the predicted and true values.
  - **Backpropagation**: The notebook includes a custom `backward_propagation` function to calculate the gradients of the weights and biases.
  - **Optimizer**: Gradient descent is used to update the weights and biases of the network.

-----

### 📊 Results and Analysis

The project demonstrates the neural network's performance on two distinct datasets:

  - **Blobs Dataset**: This dataset is relatively **linearly separable**. The model achieves a high accuracy of **97.5%**, showing its effectiveness on simple classification tasks.
      - The loss for the blobs dataset significantly decreases during training, starting from \~1.87 and ending at \~0.022.
  - **Moons Dataset**: This dataset is **non-linearly separable**. The model's performance on this dataset is also quite good, achieving an accuracy of **85.0%**. This shows that the neural network, even with a simple architecture, can learn complex decision boundaries.
      - The loss for the moons dataset also decreases, starting from \~0.32 and reaching \~0.1.

The visualizations clearly show the separation achieved by the model's predictions for both datasets.
