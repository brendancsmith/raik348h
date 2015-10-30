#!/usr/bin/env python3
# encoding: utf-8

# ----- standard modules -----

import sys

# ----- libraries -----

from sklearn.linear_model import LogisticRegression
import pandas
import numpy

# ----- -----


def regress(sample):  # dataframe
    model = LogisticRegression()

    labelColumn = 'WINETAST'

    # print(sample.columns)

    # get feature/predictor matrix as numpy array
    x = sample.drop(labelColumn, axis=1)

    x.drop('CUSTID', axis=1, inplace=True)

    x.replace('.', numpy.nan, inplace=True)
    # x.fillna(x.mean(), inplace=True)
    # x.fillna(0, inplace=True)

    print(x['SATISWWW'].map(type).unique())

    # print(x['SATISWWW'])
    print(x['SATISWWW'].mean())

    # get labels array
    y = sample[labelColumn]

    model.fit(x, y)

    model.transform(x)

    # examine the coefficients
    coef = pandas.DataFrame(list(zip(x.columns, numpy.transpose(model.coef_))))
    print(coef)


if __name__ == '__main__':
    filename = sys.argv[1]
    sample = pandas.read_csv(filename)

    regress(sample)
