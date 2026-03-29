import pandas as pd
from sqlalchemy import create_engine

#opening a connection to the database
engine = create_engine(
    "postgresql://grab:grab123@postgres:5432/grabdb"
)

#pull all data from the orders table in Postgres into memory
df = pd.read_sql("SELECT * FROM orders", engine)

#create a new column, which orderes divided by drivers online (driver efficiency)
df["orders_per_driver"] = df["orders"] / df["driver_online"]

#save the enriched data as a new table called orders_transformed
df.to_sql("orders_transformed", engine, if_exists="replace", index=False)

print("Transformation complete")