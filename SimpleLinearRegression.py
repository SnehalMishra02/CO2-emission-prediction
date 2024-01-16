import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

DataFrame = pd.read_csv("FuelConsumptionCo2.csv")
print(DataFrame.head())

df = DataFrame[['ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]

df.hist()
plt.show()

plt.scatter(df.ENGINESIZE,df.CO2EMISSIONS,color='blue')
plt.show()

mask = np.random.rand(len(df)) <0.8
train = df[mask]
test = df[~mask]


#Simple Linear Regression model of Engine Size Vs CO2 Emissions 
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print("Coeffs: ",regr.coef_)
print("Intercept: ",regr.intercept_)

plt.scatter(train_x,train_y)
plt.plot(train_x, regr.coef_[0][0]*train_x+regr.intercept_[0],'-r')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")
plt.show()

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)


MAE = np.mean(np.absolute(test_y-test_y_))
MSE = np.mean((test_y_-test_y)**2)
r2 = r2_score(test_x,test_y)

print("Mean Absolute Error: %.2f"%MAE)
print("Mean Squared Error: %.2f"%MSE)
print("R2 Score: %.2f \n\n"%r2)

#Simple Linear Regression model of Fuel Consumption Combination Vs CO2 Emissions
regr2 = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
regr2.fit(train_x,train_y)
print("Coeffs: ",regr2.coef_)
print("Intercept: ",regr2.intercept_)

plt.scatter(train_x,train_y)
plt.plot(train_x,regr2.coef_[0][0]*train_x+regr2.intercept_,'-r')
plt.xlabel("Fuel Consumption Combination")
plt.ylabel("CO2 Emissions")
plt.show()

test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y_ = regr2.predict(test_x)

MAE = np.mean(np.absolute(test_y-test_y_))
MSE = np.mean((test_y_-test_y)**2)
r2 = r2_score(test_x,test_y)

print("Mean Absolute Error: %.2f"%MAE)
print("Mean Squared Error: %.2f"%MSE)
print("R2 Score: %.2f \n\n"%r2)
