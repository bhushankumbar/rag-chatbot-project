import sqlite3
import pandas as pd
conn = sqlite3.connect("netflix.db")
df = pd.read_sql_query("SELECT * FROM raw_titles", conn)
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df = df.dropna(subset=['date_added', 'duration'])
df.to_sql("clean_titles", conn, if_exists="replace", index=False)
conn.close()
print("Success: Cleaned data pushed back into table clean_titles!")
