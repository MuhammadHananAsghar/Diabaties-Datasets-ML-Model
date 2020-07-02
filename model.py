# DIABATES DataSet From SKLEARN
# MUHMMAD HANAN ASGHAR 1ST MODEL


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabates = datasets.load_diabetes()

diabates_X = diabates.data
# print(diabates.DESCR)
# print(diabates_X)
# print(diabates.data)
# print(diabates.DESCR)
diabates_X_train = diabates_X[:-30]
diabates_X_test = diabates_X[-30:]
# print(diabates.DESCR)
diabates_Y_train = diabates.target[:-30]
diabates_Y_test = diabates.target[-30:]


model = linear_model.LinearRegression()
model.fit(diabates_X_train,diabates_Y_train)
diabates_y_predicted = model.predict(diabates_X_test)
print("Mean Squared Error : ",mean_squared_error(diabates_Y_test,diabates_y_predicted))
print("Weights: ",model.coef_)
print("Intercept: ",model.intercept_)

# Mean Squared Error :  3035.0601152912686
# Weights:  [941.43097333]
# Intercept:  153.39713623331698
