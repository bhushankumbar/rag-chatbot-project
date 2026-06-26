import pandas as pd
import sqlite3
df = pd.read_csv("netflix_titles.csv")
conn = sqlite3.connect("netflix.db")
df.to_sql("raw_titles", conn, if_exists="replace", index=False)
conn.close()
print("Success: Raw data successfully pushed to SQLite database!")
