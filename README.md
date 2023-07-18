# Solving Differential Equations Using Neural Networks

This Python code provides a framework for solving non-linear or linear differential equations using neural networks, specifically using the PyTorch library. Additionally, it includes a class that allows for function inversion using neural networks.

## *Class_EQD_torch_O1_O2*
The main component is the "*Class_EQD_torch_O1_O2*", which enables the solution of 1st or 2nd order differential equations with user-defined boundary conditions on the function and/or its derivative. To solve the equation $\psi(x,u,u',u'')=0$ with boundary conditions $(x_i,u_i)$, we employ a simple neural network with the following loss function:

$$loss=(\psi(x,u,u',u''))^2+\sum_i(u(x_i)-u_i)^2$$

Here, $u(x_i)$ represents the network's prediction at the given boundary conditions. It is worth noting that alternative boundary conditions, such as $(x_i,u'_i)$, can be chosen, allowing for the calculation of $u'(x_i)$.

## *Class_Inverse_Function*
The "*Class_Inverse_Function*" implements a straightforward method for function inversion, utilizing both function values and their corresponding pre-images.

## *Test.py*
The "*Test.py*" file contains various test cases for this research, covering 1st and 2nd order differential equations (both linear and non-linear). It also includes test cases for function inversion and explores specific equations related to the calibration function.

Please note that the code is meant for educational and research purposes, providing a starting point for solving differential equations using neural networks.
