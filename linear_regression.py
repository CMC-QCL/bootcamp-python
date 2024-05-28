import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the World Happiness data
df = pd.read_csv('./data/2019.csv')

# Choose the columns you want to use in the regression analysis
# For example, let's use 'GDP per Capita', 'Social support', 'Healthy life expectancy'
features = ['GDP per capita', 'Social support', 'Healthy life expectancy']
X = df[features]
y = df['Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression object
regressor = LinearRegression()  

# Train the model using the training sets
regressor.fit(X_train, y_train)

# Now the model is trained, and you can use it to make predictions
y_pred = regressor.predict(X_test)

# You can print out the coefficients and the mean squared error to see how well the model performed
print('Coefficients: ', regressor.coef_)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# Now the model is trained, and you can use it to make predictions
y_pred = regressor.predict(X_test)

# Plotting the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Value vs Predicted Value")

# Plotting a line for perfect correlation. The closer the scatter plot is to this line, the better the predictions.
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

plt.show()