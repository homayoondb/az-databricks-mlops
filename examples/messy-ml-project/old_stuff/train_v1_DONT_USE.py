# OLD MODEL - DO NOT USE
# keeping for reference only
# linear regression was terrible, switched to random forest

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/houses.csv")
X = df[["sqft", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)
print(f"score: {model.score(X, y)}")
# score was like 0.89 which sounds good but predictions were way off
# for expensive houses... ahmed said try random forest
