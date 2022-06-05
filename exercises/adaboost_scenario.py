import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
from IMLearn.metrics import loss_functions
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import decision_surface


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_error, test_error = np.zeros(n_learners), np.zeros(n_learners)
    for i in range(n_learners):
        train_error[i] = adaboost.partial_loss(train_X, train_y, i + 1)
        test_error[i] = adaboost.partial_loss(test_X, test_y, i + 1)

    x_range = np.arange(1, n_learners + 1)
    fig1 = go.Figure()
    fig1.add_traces([go.Scatter(x=x_range, y=train_error, name="Training Error", mode="lines", line=dict(color="blue")),
                     go.Scatter(x=x_range, y=test_error, name="Testing Error", mode="lines", line=dict(color="red"))])
    fig1.update_layout(height=600, width=950, title="loss to decision stump", xaxis_title="decision stumps",
                       yaxis_title="loss")
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} learners" for i in T],
                         horizontal_spacing=0.02, vertical_spacing=0.04)
    for i, j in enumerate(T):
        fig2.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, j), lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black")))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(height=600, width=950,
                       title=f"<b>AdaBoost: Decision Boundaries on Test Set with Noise = {noise}</b>",
                       margin=dict(t=100),
                       yaxis1_range=[-1, 1], yaxis2_range=[-1, 1], yaxis3_range=[-1, 1], yaxis4_range=[-1, 1],
                       xaxis1_range=[-1, 1], xaxis2_range=[-1, 1], xaxis3_range=[-1, 1], xaxis4_range=[-1, 1]) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    T = np.argmin(test_error) + 1

    curr_acc = loss_functions.accuracy(adaboost.partial_predict(test_X, T), test_y)
    fig3 = go.Figure([decision_surface(lambda x: adaboost.partial_predict(x, T), lims[0], lims[1], showscale=False),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    fig3.update_layout(height=600, width=950,
                       title=dict(text=f'best classifier - number of classifiers: {T}, accuracy:{curr_acc}'))
    fig3.update_xaxes(range=[-1, 1], visible=False)
    fig3.update_yaxes(range=[-1, 1], visible=False)
    fig3.show()

    # Question 4: Decision surface with weighted samples
    norm_d = adaboost.D_ / np.max(adaboost.D_) * 5
    fig4 = go.Figure([decision_surface(lambda x: adaboost.predict(x), lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(color=train_y, size=norm_d, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black")))],
                     layout=go.Layout(height=600, width=950,
                                      title=dict(text=f'classifiers - size proportional to weights')))
    fig4.update_xaxes(visible=False).update_yaxes(visible=False)
    fig4.update_xaxes(title="", range=[-1, 1], visible=False)
    fig4.update_yaxes(range=[-1, 1], visible=False)
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # Question 5
    fit_and_evaluate_adaboost(noise=0.4, n_learners=250, train_size=5000, test_size=500)
