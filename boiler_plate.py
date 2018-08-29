# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('INFY Historical Data - INFY Historical Data.csv')

#Taking 100 values
x = dataset.iloc[:100, 1:2].values   # independent
y = dataset.iloc[:100, -1].values  # dependent

# split into training dataset and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
