# forest_Fire_Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('forestfires.csv')

# Display the first few rows of the dataset
print(df.head(10))

"""X - x-axis spatial coordinate within the Montesinho park map: 1 to 9

Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9

month - month of the year: 'jan' to 'dec'

day - day of the week: 'mon' to 'sun'

FFMC - FFMC index from the FWI system: 18.7 to 96.20

DMC - DMC index from the FWI system: 1.1 to 291.3

DC - DC index from the FWI system: 7.9 to 860.6

ISI - ISI index from the FWI system: 0.0 to 56.10

temp - temperature in Celsius degrees: 2.2 to 33.30

RH - relative humidity in %: 15.0 to 100

wind - wind speed in km/h: 0.40 to 9.40

rain - outside rain in mm/m2 : 0.0 to 6.4

area - the burned area of the forest (in ha): 0.00 to 1090.84


"""

# Encoding categorical variables
le = LabelEncoder()
df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])

# Checking for missing values
print(df.isnull().sum())

# Descriptive statistics
print(df.describe())

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot for visualizing relationships
sns.pairplot(df)
plt.show()

# Histograms of all numeric columns
df.hist(figsize=(12, 10), bins=20)
plt.show()

# Boxplots to check for outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, orient="h", palette="Set2")
plt.title('Boxplots of Features')
plt.show()

# Feature Selection and Scaling
X = df.drop(['area'], axis=1)  # Features
y = np.log1p(df['area'])  # Target (log-transformed)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

"""# Model Training with Regularization"""

# Ridge Regression (L2 regularization)
ridge = Ridge()
ridge.fit(X_train_poly, y_train)
y_pred_ridge = ridge.predict(X_test_poly)

# Lasso Regression (L1 regularization)
lasso = Lasso()
lasso.fit(X_train_poly, y_train)
y_pred_lasso = lasso.predict(X_test_poly)

"""# Model Evaluation"""

# Ridge Regression Evaluation
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f'Ridge Regression - MSE: {mse_ridge}, R²: {r2_ridge}')

# Lasso Regression Evaluation
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f'Lasso Regression - MSE: {mse_lasso}, R²: {r2_lasso}')

# Hyperparameter Tuning using GridSearchCV
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search_ridge = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_ridge.fit(X_train_poly, y_train)
best_ridge = grid_search_ridge.best_estimator_

grid_search_lasso = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train_poly, y_train)
best_lasso = grid_search_lasso.best_estimator_

# Evaluate the tuned models
y_pred_best_ridge = best_ridge.predict(X_test_poly)
y_pred_best_lasso = best_lasso.predict(X_test_poly)

mse_best_ridge = mean_squared_error(y_test, y_pred_best_ridge)
r2_best_ridge = r2_score(y_test, y_pred_best_ridge)
print(f'Tuned Ridge Regression - MSE: {mse_best_ridge}, R²: {r2_best_ridge}')

mse_best_lasso = mean_squared_error(y_test, y_pred_best_lasso)
r2_best_lasso = r2_score(y_test, y_pred_best_lasso)
print(f'Tuned Lasso Regression - MSE: {mse_best_lasso}, R²: {r2_best_lasso}')

# Cross-Validation Score
cv_scores = cross_val_score(LinearRegression(), X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-Validation R² Scores: {cv_scores}')
print(f'Mean Cross-Validation R²: {np.mean(cv_scores)}')

# Scatter plot of predicted vs actual values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_ridge, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Tuned Ridge Regression)')
plt.show()

# Density plot of the residuals
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred_best_ridge, kde=True)
plt.title('Density Plot of Residuals (Tuned Ridge Regression)')
plt.show()

# Classification Accuracy for an accuracy-like measure
threshold = y_test.median()
y_test_class = y_test > threshold
y_pred_class = y_pred_best_ridge > threshold

classification_accuracy = np.mean(y_test_class == y_pred_class)
print(f'Classification Accuracy: {classification_accuracy
