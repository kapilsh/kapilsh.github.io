---
title: Gram-Schmidt Procedure
description: >-
  Solving Regression using Gram-Schmidt Procedure
date: 2018-01-07
categories: [Blog, Tutorial]
tags: [Python, Machine Learning]
pin: false
math: true
author: ks
---

An interesting way to understand [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) is [Gram-Schmidt Method](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) of successive projections to calculate the coefficients of regression. Gram-Schmidt procedure transforms the variables into a new set of orthogonal or uncorrelated variables. On applying the procedure, we should get exactly the same regression coefficients as with projection of predicted variable on the feature space.

Linear Model with number of inputs (p) such $p > 1$ is called multiple regression. We can represent least squares estimates of multiple regression in terms of estimates of univariate linear model. To understand this, let us assume a multivariate $(p > 1)$ linear model - $\mathbf{Y} = \mathbf{X}\beta + \epsilon$.

The least square estimate and residuals are:

\begin{equation}
\hat \beta = \dfrac{\sum_{n=1}^{N} x_i y_i}{\sum_{n=1}^{N} x_i^2 }
\end{equation}

and

\begin{equation}
r_i = y_i - x_i \hat \beta
\end{equation}

In convenient vector notation, we let $\mathbf{y} = (y_1, ..., y_N)^\intercal$, $\mathbf{x} = (x_1, ..., x_N)^\intercal$ and define:

\begin{equation}
\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{n=1}^{N} x_i y_i = \mathbf{x}^\intercal \mathbf{y}
\end{equation}

Hence, we can write the parameters in terms of inner product of x and y.

\begin{equation}
\hat \beta = \dfrac{\langle \mathbf{x}, \mathbf{y} \rangle}{\langle \mathbf{x}, \mathbf{x} \rangle};
\mathbf{r} = \mathbf{y} - \mathbf{x} \hat \beta
\end{equation}


> The inner product notation generalizes the linear regression to different metric spaces, as well as to probability spaces.


If the inputs $\mathbf{x_1}, \mathbf{x_2}, ..., \mathbf{x_p}$ are orthogonal, i.e. $\langle \mathbf{x_j}, \mathbf{x_k} \rangle = 0$ for all $j \neq k$, then it is easy to check that the multiple least squares estimates $\beta_j$ are equal to $\langle \mathbf{x_j}, \mathbf{y} \rangle / \langle \mathbf{x_j}, \mathbf{x_j} \rangle$.

Now if we have an intercept and a single input $\mathbf{x}$, we can find that

\begin{equation}
\hat \beta_1 = \dfrac{\langle \mathbf{x} - \bar x \mathbb{1}, \mathbf{y} \rangle}{\langle \mathbf{x} - \bar x \mathbb{1}, \mathbf{x} - \bar x \mathbb{1} \rangle}
\end{equation}

where $\bar x = \sum_{n=1}^{N} x_i / N$ and $\mathbb{1} = \mathbf{x}_0$, the vector of N ones.

The steps to generate the regression using this procedure -

1. Regress $\mathbf{x}$ on $\mathbb{1}$ to produce the residual $\mathbf{z} = \mathbf{x} - \bar x \mathbb{1}$
2. Regress $\mathbf{y}$ on the residual $\mathbf{z}$ to give the coefficient $\hat \beta_1$

where, "regress $\mathbf{b}$ on $\mathbf{a}$" means a single univariate regression of $\mathbf{b}$ on $\mathbf{a}$ with no intercept.

This process also generalizes to $p$ points and is called **Gram - Schmidt Process**. It can be understood as a process of *Successive orthogonalization* of the inputs, starting from $\mathbb{1}$.

# Gram-Schmidt Process

**ALGORITHM**

> - Initialize $\mathbf{z_0} = \mathbf{x_0} = \mathbb{1}$.

> - For all $\mathbf{x_j}$ s.t. $j$ in $\{1, 2, 3, ..., p\}$ for $p$ inputs, regress $\mathbf{x_j}$ on the residuals after $j_th$ step, where the coefficients $\hat \gamma_{lj}$ are:

\begin{equation}
\hat \gamma_{lj} = \dfrac{\langle \mathbf{z_l}, \mathbf{x_j} \rangle}{\langle \mathbf{z_l}, \mathbf{z_l} \rangle}
\end{equation}

> and residuals at each step are:

\begin{equation}
\mathbf{z_j} = \mathbf{x_j} - \sum_{k=0}^{j-1} \hat \gamma_{kj}\mathbf{z_k}
\end{equation}

> - Finally, we can calculate $\hat \beta_p$ as:

\begin{equation}
\hat \beta_p = \dfrac{\langle \mathbf{z_p}, \mathbf{y} \rangle}{\langle \mathbf{z_p}, \mathbf{z_p} \rangle}
\end{equation}

Let us test this procedure with $p = 2$.

As a first step, we will run a Multiple Regression over a set of inputs and get the regression coefficients.


```python
import numpy as np
from sklearn.linear_model import LinearRegression
```


```python
x1 = np.array([2, 2.2, 3.2, 4.5, 5.0])
x2 = np.array([45.0, 20.0, 30.0, 10.0, 6.5])
x0 = np.ones(len(x1))
z0 = x0
y = np.array([2.3, 4.5, 6.7, 8.9, 10.11])

X = np.matrix([x0, x1, x2]).T
Y = np.matrix(y).T

lin_reg = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)
lin_reg.fit(X, Y)
coeffs = lin_reg.coef_
print("Coefficients: {}".format(coeffs))
```

    Coefficients: [[ 1.39782326  1.83576285 -0.04935882]]


Now, let us calculate the $\beta_2$ using the iterative procedure.

### Step 1


```python
gamma_01 = z0.dot(x1) / (z0.dot(z0))
z1 = x1 - gamma_01 * z0
```

### Step 2:


```python
gamma_02 = z0.dot(x2) / (z0.dot(z0))
gamma_12 = z1.dot(x2) / (z1.dot(z1))
z2 = x2 - gamma_02 * z0 - gamma_12 * z1
```

### Step 3:


```python
beta_p = z2.dot(y) / (z2.dot(z2))
print(beta_p)
```

    -0.0493588203647


Similarly, we can calculate the $\beta_1$ using the iterative procedure.


```python
gamma_01 = z0.dot(x2) / (z0.dot(z0))
z1 = x2 - gamma_01 * z0

gamma_02 = z0.dot(x1) / (z0.dot(z0))
gamma_12 = z1.dot(x1) / (z1.dot(z1))
z2 = x1 - gamma_02 * z0 - gamma_12 * z1

beta_p = z2.dot(y) / (z2.dot(z2))
print(beta_p)
```

    1.83576285118


As we can see that both $\beta_1$ and $\beta_2$ match the regression coefficients obtained via L2 - norm minimization.

We can represent the transformations on $\mathbf{Z}$ more generally as:

\begin{equation}
\mathbf{X} = \mathbf{Z} \mathbf{\Gamma}
\end{equation}

where $\mathbf{z_j}$ are the columns of $\mathbf{Z}$ and $\mathbf{Gamma}$ is an upper trangular matrix with the coefficients $\gamma_{lj}$.

For $p=2$ case, we have:

\begin{equation}
\mathbf{\Gamma} =
\begin{bmatrix}
1 & \hat \gamma_{01} & \hat \gamma_{02} \\
0 & 1 & \hat \gamma_{12} \\
0 & 0 & 1
\end{bmatrix}
\end{equation}

This is similar to **QR Decomposition**. We can do a scaled QR-decomposition as:

\begin{equation}
\mathbf{X} = \mathbf{Z} \mathbf{D^{-1}} \mathbf{D} \mathbf{\Gamma} = \mathbf{Q} \mathbf{R}
\end{equation}

where,

$D_{jj} = ||\mathbf{z_j}||$, and
$\mathbf{Q^\intercal} = \mathbf{Q^{-1}}$

From here, we can calculate $\hat \beta$:

\begin{equation}
\hat \beta = (\mathbf{X^\intercal} \mathbf{X})^{-1} \mathbf{X^\intercal} y \\\
= (\mathbf{(QR)^\intercal} \mathbf{QR})^{-1} \mathbf{(QR)^\intercal} y \\\
= (\mathbf{R^\intercal} \mathbf{Q^\intercal} \mathbf{Q} \mathbf{R})^{-1} \mathbf{R^\intercal} \mathbf{Q^\intercal} y \\\
= (\mathbf{R^\intercal} \mathbf{I} \mathbf{R})^{-1} \mathbf{R^\intercal} \mathbf{Q^\intercal} y \\\
= (\mathbf{R^\intercal} \mathbf{R})^{-1} \mathbf{R^\intercal} \mathbf{Q^\intercal} y \\\
= \mathbf{R^{-1}} (\mathbf{R^\intercal})^{-1} \mathbf{R^\intercal} \mathbf{Q^\intercal} y \\\
= \mathbf{R^{-1}}  \mathbf{Q^\intercal} y \\\
\end{equation}


and:


\begin{equation}
\hat {\mathbf{y}} = \mathbf{X} \hat \beta \\\
= \mathbf{Q} \mathbf{R} \mathbf{R^{-1}} \mathbf{Q^\intercal} y \\\
= \mathbf{Q} \mathbf{Q^\intercal} y \\\
\end{equation}




#### Sources

1. [Elements of Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576)
