# House Price Prediction using Linear Regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline



df = pd.read_csv('data.csv') 


print("Data Overview:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nUnique Values in Each Column:")
print(df.nunique())




label_cols = ['street', 'city', 'statezip', 'country']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


features = [
    'bedrooms', 'bathrooms', 'floors', 'condition', 'view',
    'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
    'street', 'city', 'statezip', 'country', 'yr_built'
]

X = df[features]
y = df['price']


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
 
comparison_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print("\nComparison of Actual and Predicted Prices:")
print(comparison_df.head(10))



plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Price", color="blue", alpha=0.7)
plt.plot(y_pred, label="Predicted Price", color="red", alpha=0.7)
plt.title("Actual vs Predicted Prices")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.show()
 

plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")   
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['sqft_living'], y=df['price'])
plt.title("sqft_living vs Price")
plt.xlabel("sqft_living")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['bedrooms'], y=df['price'])
plt.title("Price Distribution by Number of Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x=df['bedrooms'])
plt.title("Count of Houses by Number of Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x=df['bathrooms'])
plt.title("Count of Houses by Number of Bathrooms")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Count")
plt.show()


print(f"📈 R² Score: {r2:.4f}")
print(f"📊 Mean Absolute Error: ₹{mae:,.2f}")
print(f"📊 Mean Squared Error: ₹{mse:,.2f}")
print(f"📊 Root Mean Squared Error: ₹{rmse:,.2f}")

 