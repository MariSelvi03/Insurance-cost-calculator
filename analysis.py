import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')

dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
dataset['smoker'] = dataset['smoker'].map({'yes': 1, 'no': 0})
dataset = pd.get_dummies(dataset, columns=['region'], drop_first=True)

sns.boxplot(x=dataset['expenses'])
plt.show()

X = dataset.drop('expenses', axis=1)
y = dataset['expenses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Linear Regression MAE: {mae:.2f} expenses')

plt.scatter(y_test, predictions)
plt.xlabel('True Values (expenses)')
plt.ylabel('Predictions (expenses)')
plt.title('True Values vs Predictions')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, 'r')
plt.show()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=0)

scaler_poly = StandardScaler()
X_train_poly = scaler_poly.fit_transform(X_train_poly)
X_test_poly = scaler_poly.transform(X_test_poly)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_poly)
predictions_poly = model_poly.predict(X_test_poly)
mae_poly = mean_absolute_error(y_test_poly, predictions_poly)
print(f'Polynomial Linear Regression MAE: {mae_poly:.2f} expenses')

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train_poly)
ridge_predictions = ridge.predict(X_test_poly)
ridge_mae = mean_absolute_error(y_test_poly, ridge_predictions)
print(f'Ridge Regression MAE: {ridge_mae:.2f} expenses')

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_poly, y_train_poly)
lasso_predictions = lasso.predict(X_test_poly)
lasso_mae = mean_absolute_error(y_test_poly, lasso_predictions)
print(f'Lasso Regression MAE: {lasso_mae:.2f} expenses')

plt.scatter(y_test_poly, ridge_predictions)
plt.xlabel('True Values (expenses)')
plt.ylabel('Ridge Predictions (expenses)')
plt.title('True Values vs Ridge Predictions')
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, 'r')
plt.show()

plt.scatter(y_test_poly, lasso_predictions)
plt.xlabel('True Values (expenses)')
plt.ylabel('Lasso Predictions (expenses)')
plt.title('True Values vs Lasso Predictions')
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, 'r')
plt.show()
