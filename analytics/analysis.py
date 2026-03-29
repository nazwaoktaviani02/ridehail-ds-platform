import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("postgresql://grab:grab123@postgres:5432/grabdb")
df = pd.read_sql("SELECT * FROM orders", engine)

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6])
df["orders_per_driver"] = df["orders"] / df["driver_online"]

print("=" * 50)
print("GRAB ORDERS — ANALYTICS REPORT")
print("=" * 50)

print("\n[1] Average orders by city")
print(df.groupby("city")["orders"].agg(["mean", "sum", "count"]).round(1))

print("\n[2] Promo impact on orders")
promo = df.groupby("promo")["orders"].mean()
lift = ((promo[1] - promo[0]) / promo[0] * 100) if 0 in promo and 1 in promo else 0
print(promo.round(1))
print(f"  --> Promo lift: {lift:.1f}%")

print("\n[3] Weather impact")
print(df.groupby("weather")["orders"].mean().sort_values(ascending=False).round(1))

print("\n[4] Driver supply vs orders correlation")
corr = df["driver_online"].corr(df["orders"])
print(f"  Pearson correlation: {corr:.4f}")

print("\n[5] Weekend vs weekday")
print(df.groupby("is_weekend")["orders"].mean().rename({True: "Weekend", False: "Weekday"}).round(1))

print("\n[6] Orders per driver efficiency by city")
print(df.groupby("city")["orders_per_driver"].mean().sort_values(ascending=False).round(3))

print("\n[7] Top 5 highest order days")
print(df.nlargest(5, "orders")[["date", "city", "orders", "promo", "weather"]].to_string(index=False))

print("\n[8] Growth trend (orders over time)")
monthly = df.groupby("month")["orders"].mean().round(1)
print(monthly)