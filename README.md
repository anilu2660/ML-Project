# ML-Project

🏡 House Price Prediction using XGBoost
This project focuses on building a machine learning pipeline to predict house prices using a dataset of residential properties. The model leverages feature engineering, data preprocessing, and the powerful XGBoost Regressor for accurate price estimation.

📁 Project Structure
app.py - The main script that performs data loading, preprocessing, model training, prediction, evaluation, and visualization.

📊 Dataset
The dataset (data.csv) includes features such as:

Property size (sqft_living, sqft_basement, etc.)

Location (city, statezip, country)

Property attributes (bedrooms, bathrooms, etc.)

🔧 Features & Engineering
The following preprocessing and feature engineering steps are applied:

Dropping irrelevant features: sqft_lot, condition, yr_built, etc.

Creating new features:

price_per_sqft

total_rooms

is_basement

living_to_above_ratio

Log transformation of the target (price) to handle skewness.

🧠 Model
A machine learning pipeline is created using:

ColumnTransformer with StandardScaler and OneHotEncoder

XGBRegressor with tuned hyperparameters

Metrics Used:
R² Score

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

📈 Visualizations
Feature correlation heatmap

Price distribution histogram

Scatter plots & box plots to understand feature impact

Actual vs Predicted price comparison (line chart & bar chart)

🔍 Results Snapshot
R² Score: High predictive power

MAE, RMSE, and MSE provide concrete error measurements

Visual comparison of actual vs predicted prices helps validate model performance




📌 Author
Developed by Anuj Upadhyay — a B.Tech CSE student passionate about AI and real-world applications of data science.



