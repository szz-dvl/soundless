import matplotlib.pyplot as plt
import pandas as pd
import math

from modules.db_se import Db

db = Db()

df = db.toDataFrame()

print(df.columns[df.isna().any()].tolist())

# Impute NaN values for BMI column
def computeBMI(row: pd.Series):
    weight = row["weight"]
    height = row["height"] * 12

    return (weight / height ** 2) * 703

def perRow(row: pd.Series):
    if math.isnan(row["bmi"]):
        row["bmi"] = computeBMI(row)
    return row

df = df.apply(perRow, axis=1)


fig, ax = plt.subplots()
ax.set_ylabel('Sleep Efficiency')

bplot = ax.boxplot(df["se"],
                   patch_artist=True)  # will be used to label x-ticks
ax.set_xticks([])
plt.grid(visible=True)
plt.show()

fig, ax = plt.subplots()
ax.set_ylabel('ESS')

bplot = ax.boxplot(df["ess"],
                   patch_artist=True)  # will be used to label x-ticks
ax.set_xticks([])
plt.grid(visible=True)
plt.show()