    
# IMPORT LIBRARIES
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA


# LOAD DATA
# ======================================
df = pd.read_csv("house_data.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# ENCODE LOCATION
# ======================================
le = LabelEncoder()
df["location_encoded"] = le.fit_transform(df["location"])


# FEATURES & TARGET
# ======================================
X = df[["area","bedrooms","bathrooms","floors","parking","location_encoded","Year"]]
y = df["price"]


# TRAIN RANDOM FOREST
# ======================================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)


# SELECT SAMPLE DATA (15 ROWS)
# ======================================
sample = df.sample(8, random_state=1).reset_index(drop=True)

X_sample = sample[["area","bedrooms","bathrooms","floors","parking","location_encoded","Year"]]


# PREDICTIONS
# ======================================
actual = sample["price"].values
predicted = rf.predict(X_sample)


# ACCURACY METRICS
# ======================================
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print("\n================ MODEL PERFORMANCE ================\n")

# MAE
print("MAE Formula:")
print("MAE = (1/n) * Σ |Actual - Predicted|\n")

print("MAE Calculation:")
for i in range(len(actual)):
    print(f"|{int(actual[i])} - {int(predicted[i])}| = {int(abs(actual[i] - predicted[i]))}")

print("\nFinal MAE:", int(mae))


# R2 Score
print("\nR2 Formula:")
print("R² = 1 - [ Σ(Actual - Predicted)² / Σ(Actual - Mean)² ]\n")

print("Final R2 Score:", round(r2 * 100, 2), "%")


# ARIMA (YEAR TREND)
# ======================================
yearly_avg = df.groupby("Year")["price"].mean()
yearly_avg.index = pd.PeriodIndex(yearly_avg.index, freq='Y')

model = ARIMA(yearly_avg, order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=1)

last_actual = yearly_avg.iloc[-1]
future_trend = forecast.iloc[0]

trend_growth = future_trend / last_actual


# FUTURE PRICE (ARIMA + RANDOM 5–8%)
# ======================================
random.seed(42)

future = []
for price in predicted:
    growth = random.uniform(1.05, 1.08)
    future_price = price * trend_growth * growth
    future.append(future_price)

future = np.array(future)


# FORMAT PRICE
# ======================================
def format_price(x):
    if x >= 10000000:
        return f"₹{x/10000000:.1f}Cr"
    else:
        return f"₹{x/100000:.1f}L"


# CREATE LABELS
# ======================================
labels = []
for i in range(len(sample)):
    row = sample.iloc[i]
    label = (
        f"{row['location']}\n"
        f"({row['Year']} → {row['Year'] + 1})\n"
        f"{row['area']} sqft | {row['floors']}Flr\n"
        f"{row['bathrooms']}Bath | {row['bedrooms']}Bed\n"
        f"{row['parking']}Park"
    )
    
    labels.append(label)

# BAR GRAPH
# ======================================
x = np.arange(len(labels))
width = 0.20

plt.figure(figsize=(22,10))

# Gradient Background
ax = plt.gca()

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

ax.imshow(
    gradient,
    aspect='auto',
    cmap=plt.get_cmap('cool'),
    extent=[-1, len(labels), 0, max(future)*1.1],
    alpha=0.35
)

bars1 = plt.bar(x - width, actual, width, label="Actual Price(present year)", color='blue')
bars2 = plt.bar(x, predicted, width, label="Predicted Price (predicted year)", color='orange')
bars3 = plt.bar(x + width, future, width, label="Future Price (next year)", color='green')


# ADD VALUE ON TOP
# ======================================
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            format_price(height),
            ha='center',
            va='bottom',
            fontsize=7.5
        )


# AXIS SETTINGS
# ======================================
plt.xticks(x, labels, rotation=0, ha='center')

plt.xlabel("Year, Location, Area, Bedrooms, Bathrooms, Floors, Parking")
plt.ylabel("Price")


# Y-AXIS (₹30 LAKHS GAP)
# ======================================
step = 3000000 #30Lacks

y_max = max(max(actual), max(predicted), max(future)) + step

yticks = np.arange(0, y_max, step)

plt.yticks(yticks, [format_price(y) for y in yticks])
plt.ylim(0, y_max)

# TITLE & LEGEND
# ======================================
plt.title("House Price: Actual vs Predicted vs Future")

plt.legend(loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()