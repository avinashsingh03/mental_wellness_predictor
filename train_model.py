import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load CSV
df = pd.read_csv("ScreenTime vs MentalWellness1.csv")


# Features & Target
X = df.drop(columns=["mental_wellness_index_0_100"])
y = df["mental_wellness_index_0_100"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Test MSE: {mse:.2f}")


# Save Model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# import joblib

# # Load CSV
# df = pd.read_csv("ScreenTime vs MentalWellness1.csv")

# # Features & Target
# X = df.drop(columns=["mental_wellness_index_0_100"])
# y = df["mental_wellness_index_0_100"]

# # One-hot encode categorical features
# X = pd.get_dummies(X, drop_first=True)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# preds = model.predict(X_test)
# print("Test MSE:", ((y_test - preds) ** 2).mean())

# # Save model
# joblib.dump(model, "model.pkl")
# print("Model saved as model.pkl")
