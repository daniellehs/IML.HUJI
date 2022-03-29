import math

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd

pio.templates.default = "simple_white"

MU = np.array([0, 0, 4, 0]).T
SIGMA = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

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
    fig1 = px.scatter(x=np.arange(10, 1001, 10), y=expec_arr)
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    fig2 = px.scatter(x=samples_1, y=(univar.pdf(samples_1)))
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples_2 = np.random.multivariate_normal(mean=MU, cov=SIGMA, size=1000)
    multivar = MultivariateGaussian()
    print(multivar.mu_)
    print('\n')
    print(multivar.cov_)
    print('\n')


    # Question 5 - Likelihood evaluation
    max_f1, max_f3 = 0, 0
    max_ll = -np.inf
    log_like = []
    lins_range = np.linspace(-10, 10, 200)
    for f1 in lins_range:
        curr = []
        for f3 in lins_range:
            curr_arr = np.array([f1, 0, f3, 0])
            ll = multivar.log_likelihood(curr_arr, SIGMA, samples_2)
            curr.append(ll)
            if ll > max_ll:
                max_f1, max_f3 = f1, f3
                max_ll = ll
        log_like.append(curr)
    fig = px.imshow(log_like, labels =dict(x='f3', y='f1', color='log likelyhood'), text_auto=True)
    fig.update_layout(title='log likelyhood heatmap of f1 and f3 vals')
    fig.show()


    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
