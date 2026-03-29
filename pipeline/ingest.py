import pandas as pd
from sqlalchemy import create_engine

#read the file and turn into table in memory
df = pd.read_csv("data/grab_orders.csv")

#create a connection to the Postgres db
engine = create_engine(
    "postgresql://grab:grab123@postgres:5432/grabdb"
)

"""
postgresql => protocol
grab => user
password => grab123
host => postgres
port => :5432
grabdb => database name
"""

#save the df to Postgres as a table named orders
df.to_sql("orders", engine, if_exists="replace", index=False)

print("Data successfully loaded")