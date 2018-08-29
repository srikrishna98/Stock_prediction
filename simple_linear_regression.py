# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('INFY Historical Data - INFY Historical Data.csv')

#Taking 100 values
x = dataset.iloc[:100, 1:2].values   # independent - Price
y = dataset.iloc[:100, -1].values  # dependent - Change

# split into training dataset and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#To find the change percentage of a particular price
# y_pred = regressor.predict([[Price]])


# Visualising the Training set results
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Price vs Change (Training set)')
plt.xlabel('Price')
plt.ylabel('Change')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Change (Test set)')
plt.xlabel('Price')
plt.ylabel('Change')
plt.show()
