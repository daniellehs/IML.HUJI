import math

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples_1 = np.random.normal(10, 1, 1000)
    univar = UnivariateGaussian()
    univar.fit(samples_1)
    print((univar.mu_, univar.var_))

    # Question 2 - Empirically showing sample mean is consistent
    expec_arr = np.zeros(100)
    for i in range(100):
        samples_i = samples_1[:10 * i]
        expec_arr[i] = np.mean(samples_i) - univar.mu_
    # exp_vs_real = pd.DataFrame(data=expec_arr, index=np.arange(10, 1000, 10))
    fig = px.scatter(x=np.arange(10, 1001, 10), y=expec_arr)
    fig.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
