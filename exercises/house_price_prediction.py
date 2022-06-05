from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop lines with negative prices
    neg_price_lines = df[df['price'] < 0].index
    df.drop(neg_price_lines, inplace=True)

    to_delete = pd.get_dummies(df["zipcode"])
    df = pd.concat([df, to_delete], axis=1)

    # drop the id and long columns
    df.drop(['id', 'long', 'date', 'zipcode'], axis=1, inplace=True)

    # df = pd.concat([df, Z], axis=1)
    df.dropna(inplace=True)
    p = df['price']
    d = df.drop(columns=['price'])

    return p, d


def pearson_correlation(feature, prices):
    """
    return the pearson correlation between each feature and prices vector
    """
    return (np.cov(feature, prices) / (np.std(feature) * np.std(prices)))[0][1]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X:
        pearson_cor = round(pearson_correlation(X[col], y), 3)
        figure1 = go.Figure([go.Scatter(x=X[col], y=y, fill=None, mode="markers")],
                  layout=go.Layout(title="scatter plot of " + col + " and response, pearson correlation value is "
                                         + str(pearson_cor), xaxis=dict(title="feature"), yaxis=dict(title="response")))
        figure1.show()

        # figure = px.scatter(x=X[col], y=y, title="scatter plot of" + col + "and response, pearson correlation value"
        #                                                                 "is" + str(pearson_correlation(col, X[col])))
        # figure.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    prices, design = load_data('//Users//danielle//IML.HUJI//datasets//house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(design.loc[:, :"sqft_lot15"], prices, "../")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(design, prices, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_loss = []
    std_loss = []
    samp = LinearRegression()
    for i in range(10, 101):
        temp_loss = []
        for j in range(10):
            X_train_samples = train_X.sample(frac=i / 100)
            y_train_samples = train_y.loc[X_train_samples.index]
            samp.fit(np.asarray(X_train_samples), np.asarray(y_train_samples))
            loss = samp.loss(np.asarray(test_X), np.asarray(test_y))
            temp_loss.append(loss)
        mean_loss.append(np.mean(temp_loss))
        std_loss.append(np.std(temp_loss))
    training_size = np.arange(10, 101)
    mean_loss = np.array(mean_loss)
    std_loss = np.array(std_loss)
    fig_a = go.Scatter(x=training_size, y=mean_loss, name="Mean loss", line=dict(color="red"))
    fig_b = go.Scatter(x=training_size, y=mean_loss + 2 * std_loss, fill='tonexty', mode='lines',
                      line=dict(color="lightpink"), name="top of CI")
    fig_c = go.Scatter(x=training_size, y=mean_loss - 2 * std_loss, fill='tonexty', mode='lines',
                      line=dict(color="lightpink"), name="bottom of CI")
    fig = go.Figure(fig_a, layout=go.Layout(title="Mean loss as function of p%", xaxis=dict(title="p%"),
                                           yaxis=dict(title="mean loss")))
    fig.add_traces([fig_b, fig_c])
    fig.show()
