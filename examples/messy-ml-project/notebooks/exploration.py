# just some quick exploration i did last tuesday
# TODO: clean this up later

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("data/houses.csv")
print(df.head())
print(df.describe())

# some plots
df.hist(figsize=(10,8))
plt.savefig("data/exploration.png")
print("saved plot to data/exploration.png")

# correlation
print("\nCorrelations with price:")
print(df.corr(numeric_only=True)["price"].sort_values(ascending=False))

# sqft seems most important, bedrooms too
# age_years negatively correlated - makes sense
# distance_to_city also negative - closer = more expensive
