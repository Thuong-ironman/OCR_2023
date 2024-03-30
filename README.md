# OCR 2023
Assignment 3: Learning for control

## 1 Description
Learning a Value function with a Neural Network, and using it as terminal cost in an optimal control problem

## 2 Commmon Tools

### 2.1 Solving Optimal Control Problems (OCPs)

As an interesting alternative to using the single-shooting formulation that I learned during my lab sessions, I use CasADi. CasADi is a software library for
modeling and solving OCPs, coming with Python bindings. CasADi can be easily installed using pip:

pip install casadi

### 2.2 Training Neural Networks

Code for creating and training a neural network using the Tensor Flow/Keras library is available in the file train.py.

### 2.3 Parallel Computation

To make your code run faster you can try to parallelize it, so that it can exploit multiple CPU cores. To do that, I use the Python library multiprocessing.

## 3 Project A - Learning a Terminal Cost

The aim of this project is to learn an approximate Value function that can then be used as terminal cost in an MPC formulation. The idea is similar to the one explored in the second assignment. First, we have to solve many OCP’s starting from different initial states (either chosen randomly or on a grid). For every solved OCP, we should store the initial state x_0 and the corresponding optimal cost J(x_0) in a buffer. Then, we should train a neural network to predict the optimal cost J given the initial state x_0. Once such a network has been trained, we must use it as a terminal cost inside an OCP with the same formulation, but with a shorter horizon (e.g. half). We should be able to empirically show that the introduction of the terminal cost compensates the decrease of the horizon length.