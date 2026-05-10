# Bayesian Optimization - ARM64 Assembly

This project contains a bare-metal implementation of Bayesian Optimization written entirely in ARM64 Assembly for Apple Silicon .

> **Disclaimer:** This assembly implementation was written in a burst of exploratory energy and may contain redundant or inefficient instructions. Some parts are probably suboptimal, and a few might be wrong in ways I haven’t discovered yet. It’s what happens when curiosity meets assembly and neither side wins.

## Objective

The goal of this code is to find the absolute minimum of the mathematical function:
`f(x) = sin(3x) + x^2 - 0.7*cos(5x)`
within the bounded range `x ∈ [-2, 2]`.

Instead of running a loop over thousands of points or using standard gradient descent, it implements a probabilistic search algorithm that hones in on the deepest valley with as few evaluations of `f(x)` as possible.

## Core Components

The program is self-contained and implements linear algebra and probabilistic models entirely in assembly:

- **Surrogate Model (Gaussian Process):** Builds a probabilistic model of the function. It uses a **Radial Basis Function (RBF) Kernel** to evaluate how similar any two points are (with length scale `ℓ=0.5` and noise `σ²=0.01`).
- **Matrix Factorization (Cholesky Decomposition):** In order to calculate the Gaussian Process predictions, the code mathematically solves the `K` covariance matrix by decomposing it into a lower triangular matrix `L`, doing this purely with ARM64 floating-point instructions.
- **Acquisition Function (Lower Confidence Bound - LCB):** Decides the best next point to sample. It calculates `LCB(x) = μ(x) - κ·σ(x)` where `κ=2.0`. This balances exploiting areas we know are low, against exploring areas with high uncertainty.
- **Grid Search:** A discrete optimizer that scores 200 evenly spaced points across the `[-2, 2]` grid during each iteration to find the absolute lowest LCB.

## Build and Run

To build and execute using the provided Makefile:

```bash
make run
```

Alternatively, compile directly:

```bash
clang -arch arm64 -o bayesian_opt bys.s
./bayesian_opt
```

##  Output

The code gives this as output (note: it could be wrong, verify the math yourself!):

```text
Bayesian Optimisation
f(x) = sin(3x) + x^2 - 0.7*cos(5x),  x in [-2, 2]

[iter  1]  x =   0.5930   f(x) =   2.0191
[iter  2]  x =  -0.5327   f(x) =  -0.0944
[iter  3]  x =  -0.1910   f(x) =  -0.9100
[iter  4]  x =  -0.1910   f(x) =  -0.9100
[iter  5]  x =  -0.1709   f(x) =  -0.9210
[iter  6]  x =  -0.1709   f(x) =  -0.9210
[iter  7]  x =  -0.1709   f(x) =  -0.9210
[iter  8]  x =  -0.1709   f(x) =  -0.9210
[iter  9]  x =  -0.1709   f(x) =  -0.9210
[iter 10]  x =  -0.1709   f(x) =  -0.9210
[iter 11]  x =  -0.1709   f(x) =  -0.9210
[iter 12]  x =  -0.1709   f(x) =  -0.9210
[iter 13]  x =  -0.1709   f(x) =  -0.9210
[iter 14]  x =  -0.1709   f(x) =  -0.9210
[iter 15]  x =  -0.1709   f(x) =  -0.9210

=== Best after 15 iterations ===
  x    =  -0.170854
  f(x) =  -0.920959
```
