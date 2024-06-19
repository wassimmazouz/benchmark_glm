# **Generalized Linear Models** (GLM) Benchmark
This repository is dedicated to benchmarking GLMs using the Benchopt framework.

This is a benchmark based on the Benchopt framework. You can learn more about it [here](https://benchopt.github.io/).

## **Theoretical Overview**
A generalized linear model (GLM) is a flexible generalization of ordinary linear regression. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a *link function*. In a generalized linear model, the outcome <span> $\mathbf{Y}$ </span> (dependent variable) is assumed to be generated from a particular distribution in a family of exponential distributions (e.g. Normal, Binomial, Poisson, Gamma). The mean <span> $\mathbf{\mu}$ </span> of the distribution depends on the independent variables <span> $\mathbf{X}$ </span> through the relation:

$$\mathbb{E}[\boldsymbol{Y}|\boldsymbol{X}] = \boldsymbol{\mu} = g^{-1}(\boldsymbol{X}\,\boldsymbol{\beta})$$

where <span> $\mathbb{E}[\boldsymbol{Y}|\boldsymbol{X}]$ </span> is the expected value of <span> $\boldsymbol{Y}$ </span> conditioned to <span> $\boldsymbol{X}$ </span>, <span> $\boldsymbol{X}\hspace{1pt}\boldsymbol{\beta}$ </span> is the linear predictor and <span> $g(\cdot)$ </span> is the link function.

##  **Practical Examples**
As already mentioned, let <span> $Y$ </span> be the outcome (dependent variable) and <span> $\mathbf{X}$ </span> be the independent variables. The three types of regression analyzed here(*Linear*, *Logistic* and *Poisson*) differ in the nature of $Y$. For each type, ad hoc datasets and solvers were collected.

------------------------

###  **Linear Regression**
In the case of linear regression, $Y$ is modeled as:

$$\begin{cases}
    \hspace{4pt} Y\sim N(\mu,\sigma^2)\\
    \hspace{4pt} \mu = \boldsymbol{X}\hspace{1pt}\boldsymbol{\beta}
\end{cases}$$

The following datasets are used:
* [The *bodyfat* LIBSVM dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html)
* [The *diabetes* `sklearn` dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
* [The *California housing* `sklearn` dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
* A simulated dataset

------------------------

### **Logistic Regression**
In the case of logistic regression $Y$ is a categorical value (** be sure to have values between $-1$ and $1$ **) and it is modeled as:

$$\begin{cases}
    \hspace{4pt} Y \sim Bernoulli(\mu)\\
    \hspace{4pt} \log(\frac{\mu}{1-\mu}) = \boldsymbol{X}\hspace{1pt}\boldsymbol{\beta}
\end{cases}$$

The following datasets are used :
* [The `sklearn` *breast cancer* dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
* A simulated dataset

------------------------

### **Poisson Regression**
In the case of poisson regression, $Y$ is a count value and it is modeled as:

$$\begin{cases}
    \hspace{4pt} Y \sim Poisson(\mu)\\
    \hspace{4pt}\log(\mu) = \boldsymbol{X}\hspace{1pt}\boldsymbol{\beta}
\end{cases}$$

For Poisson regression, the following datasets were used :
* [The freMTPL insurance dataset](https://www.openml.org/search?type=data&status=active&id=41214)
* A simulated dataset with different levels of sparsity for the design matrix $\boldsymbol{X}$

------------------------

## How to use this benchmark

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/wassimmazouz/benchmark_glm
   $ cd benchmark_glm
   $ benchopt run .

Options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run . -s sklearn -d bcancer --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.
