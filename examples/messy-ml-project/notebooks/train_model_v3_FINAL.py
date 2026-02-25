# house price prediction model
# v3 - this one actually works!! dont delete
# ahmed wrote v1, i rewrote it, sarah fixed the bug on line 34

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')  # lol

# load data
df = pd.read_csv("data/houses.csv")
print(f"loaded {len(df)} rows")

# features - i removed 'garage' cause it was messing things up (ask sarah why)
features = ["sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
target = "price"

X = df[features]
y = df[target]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Results ===")
print(f"MSE:  {mse:,.0f}")
print(f"MAE:  {mae:,.0f}")
print(f"R2:   {r2:.4f}")
print(f"RMSE: {mse**0.5:,.0f}")

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nsaved model to model.pkl")

# feature importance
print("\n=== Feature Importance ===")
for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")
