# Pytorch Beginer Series

- Overview of Pytorch
- Tensors
- Autograd
- Building a Model
- Loading Data
- Training models
- deploy with tochscript

PyTorch is an open-source deep learning framework developed by Facebook’s AI Research (FAIR) lab.

## Core Features

- Tensors: Multidimensional arrays (like NumPy arrays) that can run on CPU or GPU.
- Autograd: Automatic differentiation for backpropagation — PyTorch keeps track of all operations so you can get gradients without manually coding derivatives.

- Dynamic Computation Graph: Instead of pre-defining the whole model (like in TensorFlow’s older versions), PyTorch builds the computation graph on the fly — making debugging and experimentation much easier.

- nn Module: High-level neural network building blocks (torch.nn for layers, activation functions, loss functions).

- Optimizers: Pre-built optimization algorithms (SGD, Adam, etc.).

## Autograds:

It automatically computes gradients (derivatives) of tensors during the backpropagation step of training neural networks.

Instead of you manually calculating derivatives for your loss function, autograd tracks all operations on tensors and figures out the gradients for you.
