import math

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

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

    fig1 = px.scatter(x=np.arange(10, 1001, 10), y=expec_arr, labels=dict(x='sample size',
                                                                          y='distance between the estimated and true '
                                                                            'value of the expectation'))
    fig1.update_layout(
        title='distance between the estimated and true value of the expectation according to sample size')
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    fig2 = px.scatter(x=samples_1, y=(univar.pdf(samples_1)), labels=dict(x='sample size', y='pdf value'))
    fig2.update_layout(title='pdf values according to sample size')
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples_2 = np.random.multivariate_normal(mean=MU, cov=SIGMA, size=1000)
    multivar = MultivariateGaussian()
    multivar.fit(samples_2)
    print(multivar.mu_)
    print(multivar.cov_)

    # Question 5 - Likelihood evaluation
    log_like = []
    max_ll = -np.Inf
    max_f1, max_f3 = -np.Inf, -np.Inf
    lins_range = np.linspace(-10, 10, 200)
    for f1 in lins_range:
        curr = []
        for f3 in lins_range:
            curr_arr = np.array([f1, 0, f3, 0])
            ll = multivar.log_likelihood(curr_arr, SIGMA, samples_2)
            if ll > max_ll:
                max_ll = ll
                max_f1, max_f3 = f1, f3
            curr.append(ll)
        log_like.append(curr)
    fig = px.imshow(log_like, x=lins_range, y=lins_range, labels=dict(x='f3', y='f1', color='log likelyhood'),
                    text_auto=True)
    fig.update_layout(title='log likelyhood heatmap of f1 and f3 vals')
    fig.show()

    # Question 6 - Maximum likelihood
    print(round(max_f1, 3), round(max_f3, 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
