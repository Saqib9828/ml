#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:08:22 2019

@author: fist-11
"""

from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
X, _ = load_digits(return_X_y=True)
transformer = FastICA(n_components=7,
        random_state=0)
X_transformed = transformer.fit_transform(X)
print(X_transformed.shape)