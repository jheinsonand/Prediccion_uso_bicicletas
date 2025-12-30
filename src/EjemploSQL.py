from database import load_raw_from_github_to_db, read_data_sql

# Solo la primera vez:
load_raw_from_github_to_db()

# Luego, para leer:
query = """
SELECT 
    date_time,
    users,
    temp,
    hum
FROM bike_sharing
ORDER BY date_time
"""
df = read_data_sql(query)
print(df.head())