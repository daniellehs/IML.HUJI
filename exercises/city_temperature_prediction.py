import plotly.express

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    data = pd.read_csv(filename, parse_dates=['Date'])
    data.drop_duplicates(inplace=True)
    neg_temp_lines = data[data['Temp'] < -70].index
    data.drop(neg_temp_lines, inplace=True)
    data['DayOfYear'] = data["Date"].dt.dayofyear
    data.dropna(inplace=True)
    return data



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('//Users//danielle//IML.HUJI//datasets//City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_data = df.loc[df['Country'] == 'Israel']
    fig1 = px.scatter(x=israel_data['DayOfYear'], y=israel_data['Temp'], color=israel_data['Year'].astype(str),
                     title="Average temperature in Israel according to day of the year",
                     labels=dict(x="day of the year", y="average temperature"))
    fig1.show()
    group_by_month = israel_data.groupby('Month').agg('std')
    fig2 = px.bar(group_by_month, x=np.arange(1, 13), y="Temp",
                  title="Standard deviation of daily temperature according to month")
    fig2.show()

    # Question 3 - Exploring differences between countries

    groups = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    fig = px.line(x=groups['Month'], y=groups['mean'], error_y=groups['std'], color=groups['Country'],
            title="Average temperature in different countries by month", labels=dict(x='Month', y='Average Temp',
                                                                                 color='Country'))
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_data['DayOfYear'], israel_data['Temp'])
    loss = []
    for i in range(1, 11):
        samp = PolynomialFitting(i)
        samp.fit(np.asarray(train_X), np.asarray(train_y))
        curr_loss = samp.loss(np.asarray(test_X), np.asarray(test_y))
        loss.append(round(curr_loss, 2))
    for i in range(10):
        print("k = " + str(i+1) + ", error = " + str(loss[i]))
    fig = px.bar(x=np.arange(1, 11), y=np.array(loss), title="Test error recorded according to k value",
                 labels=dict(x="k value", y="test error"))
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    opt_k = np.argmin(np.array(loss)) + 1
    samp2 = PolynomialFitting(opt_k)
    samp2.fit(np.asarray(israel_data['DayOfYear']), np.asarray(israel_data['Temp']))
    pred = []
    south_africa_d = df.loc[df["Country"] == "South Africa"]
    jordan_d = df.loc[df["Country"] == "Jordan"]
    nether_d = df.loc[df["Country"] == "The Netherlands"]
    countries_data = [south_africa_d, jordan_d, nether_d]
    for i in range(3):
        curr_country = countries_data[i]
        pred.append(samp2.loss(np.asarray(curr_country["DayOfYear"]), np.asarray(curr_country['Temp'])))
    fig = px.bar(x=["South Africa", "Jordan", "The Netherlands"], y=pred,
                 title="Model's error over each of the other countries", labels=dict(x="Country", y="Model's error"))
    fig.show()
