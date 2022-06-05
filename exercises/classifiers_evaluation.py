from math import atan2, pi

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy

from utils import custom

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        path_to_file = "/Users/danielle/IML.HUJI/datasets/"
        # Load dataset
        X, y = load_dataset(path_to_file + f)

        def callback(fit: Perceptron):
            losses.append(fit._loss(X, y))

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        model = Perceptron(callback=callback)

        model._fit(X, y)

        # Plot figure
        x_range = list(range(1, len(losses) + 1))
        px.line(x=x_range, y=losses, title=n, labels=dict(x="iteration", y="missclassification error")).show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")



def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    path_to_file = "/Users/danielle/IML.HUJI/datasets/"
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(path_to_file + f)

        # Fit models and predict over training set
        gnb_model = GaussianNaiveBayes()
        lda_model = LDA()
        gnb_model.fit(X, y)
        lda_model.fit(X, y)

        # Q1
        models = [gnb_model, lda_model]
        preds = [gnb_model.predict(X), lda_model.predict(X)]
        accuracies = [accuracy(y, preds[0]), accuracy(y, preds[1])]
        titles = ["GNB model estimator" + " accuracy: " + str(accuracies[0]), "LDA model estimator" + " accuracy: " +
                  str(accuracies[1])]
        symbols = np.array(["circle", "square", "star"])
        accuracies = [accuracy(y, preds[0]), accuracy(y, preds[1])]

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=titles)

        for i in range(len(models)):
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=preds[i], symbol=symbols[y],
                                                   opacity=0.65, line=dict(width=1)))], rows=1, cols=i + 1)
        fig.update_xaxes(title_text="feature 0")
        fig.update_yaxes(title_text="feature 1")
        fig.update_layout(title=f + " dataset")

        # Q2
        for i, model in enumerate(models):
            fig.add_traces([go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color='black', symbol='x',
                                                   opacity=0.65, line=dict(width=1)))], rows=1, cols=i + 1)

        # Q3
        for j in range(len(gnb_model.classes_)):
            fig.add_trace(get_ellipse(models[0].mu_[j], np.diag(models[0].vars_[j])), row=1, col=1).\
                add_trace(get_ellipse(models[1].mu_[j], models[1].cov_), row=1, col=2)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
