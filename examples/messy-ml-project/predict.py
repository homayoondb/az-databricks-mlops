# quick prediction script - run after training
# usage: python predict.py

import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# test prediction
test_house = pd.DataFrame([{
    "sqft": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 10,
    "distance_to_city_km": 15,
}])

pred = model.predict(test_house)[0]
print(f"Predicted price: ${pred:,.0f}")
