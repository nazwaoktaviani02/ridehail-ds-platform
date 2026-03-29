import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# Load Data
df = pd.read_csv("data/grab_orders.csv")

# extract useful information from it into separate columns
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek # Monday = 0, Sunday = 6
df["month"] = df["date"].dt.month # January = 1, December = 12
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int) # 1 if weekend, 0 if not

# Encode categorical
le_city = LabelEncoder()
le_weather = LabelEncoder()
df["city_encoded"] = le_city.fit_transform(df["city"])
df["weather_encoded"] = le_weather.fit_transform(df["weather"])

# define features & target
# features (X) = input the model uses to make predictions
# target (y) = what the model is trying to predict (number of orders)
features = ["promo", "driver_online", "city_encoded", "weather_encoded", "day_of_week", "is_weekend", "month"]
X = df[features]
y = df["orders"]

# split data into 80% training data and 20% testing daata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train) # <- where learning happens

# test on the 20% data the model has never been seen before
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred) # avg error in number of orders
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # penalizes big mistakes more
r2 = r2_score(y_test, y_pred) # 1.0 = perfect, 0  = useless

print("=== Model Evaluation ===")
print(f"MAE  : {mae:.2f}") # how many orders off on average
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}") # closer to 1.0 = better

# Feature importance
print("\n=== Feature Importance ===")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<20} {imp:.4f}")

# Save model and encoders
joblib.dump(model, "analytics/demand_model.pkl")
joblib.dump(le_city, "analytics/le_city.pkl")
joblib.dump(le_weather, "analytics/le_weather.pkl")
joblib.dump(features, "analytics/model_features.pkl")

print("\nModel trained and saved.")