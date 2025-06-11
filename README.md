# ML-Project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


df = pd.read_csv("data.csv")


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


df.drop(['sqft_lot', 'condition', 'yr_built', 'yr_renovated', 'street'], axis=1, inplace=True)


df['price_per_sqft'] = df['price'] / df['sqft_living']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['is_basement'] = (df['sqft_basement'] > 0).astype(int)
df['living_to_above_ratio'] = df['sqft_living'] / (df['sqft_above'] + 1)


df = df[df['price'] < 5_000_000]


y = np.log1p(df['price'])
X = df.drop('price', axis=1)


categorical_cols = ['city', 'statezip', 'country']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)


y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)


r2 = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title(" Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(np.expm1(y), bins=50, kde=True)
plt.title(" Distribution of House Prices (Actual)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['sqft_living'], y=np.expm1(y))
plt.title(" Sqft Living vs Price")
plt.xlabel("Sqft Living")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=df['bedrooms'], y=np.expm1(y))
plt.title(" Price Distribution by Number of Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(y_actual.values[:100], label="Actual Price", color="blue", alpha=0.7)
plt.plot(y_pred[:100], label="Predicted Price", color="red", alpha=0.7)
plt.title(" Actual vs Predicted House Prices")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.show()


comparison_df = pd.DataFrame({'Actual Price': y_actual[:25].values, 'Predicted Price': y_pred[:25]})

print(comparison_df)
comparison_df.plot(kind='bar', figsize=(14, 6))
plt.title(" Actual vs Predicted Prices (First 25)")
plt.xlabel("Index")
plt.ylabel("Price")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()


print(f"📈 R² Score: {r2:.4f}")
print(f"📊 MAE: ₹{mae:,.2f}")
print(f"📊 RMSE: ₹{rmse:,.2f}")
print(f"📊 MSE: ₹{mse:,.2f}")
print(f"📊 Mean Price: ₹{np.mean(np.expm1(y)):.2f}")
print(f"📊 Median Price: ₹{np.median(np.expm1(y)):.2f}")
print(f"📊 Std Dev of Price: ₹{np.std(np.expm1(y)):.2f}")





