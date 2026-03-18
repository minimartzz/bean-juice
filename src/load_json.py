"""
load_json.py — Loads raw coffee.json into DuckDB
================================================
Takes a coffee.json and extracts the entries into
a DuckDB database

Components:
- JSON_PATH: Path to raw coffee.json
- DB_PATH: Path to store database
"""
import duckdb
import os
from dotenv import load_dotenv
load_dotenv()

JSON_PATH = os.environ["JSON_PATH"]
DB_PATH   = os.environ["DB_PATH"]

con = duckdb.connect(DB_PATH)
print("=" * 20)
print(f"DuckDB created at {DB_PATH}")
print("=" * 20)

# Create the schema
# con.execute("CREATE SCHEMA IF NOT EXISTS coffee;")
con.execute("DROP TABLE IF EXISTS reviews;")

# Data ingestion
con.execute(f"""
  CREATE TABLE reviews AS
    SELECT
      (data->>'$.rating')::INTEGER AS rating,
      (data->>'$.roaster') AS roaster,
      (data->>'$.bean') AS bean,
      (data->>'$.Roaster Location') AS location,
      (data->>'$.Coffee Origin') AS origin,
      (data->>'$.Roast Level') AS roast_level,
      (data->>'$.agtron') AS agtron,
      (data->>'$.Est Price') AS est_price,
      strptime(data->>'$.Review Date', '%B %Y')::DATE AS review_date,
      (data->>'$.Aroma')::INTEGER AS aroma,
      (data->>'$.Body')::INTEGER AS body,
      (data->>'$.Flavor')::INTEGER AS flavor,
      (data->>'$.Aftertaste')::INTEGER AS aftertaste,
      (data->>'$.With Milk')::INTEGER AS with_milk,
      (data->>'$.Acidity/Structure')::INTEGER AS acid_structure,
      (data->>'$.Blind Assessment')::TEXT AS blind_assessment,
      (data->>'$.Notes')::TEXT AS notes,
      (data->>'$.Who Should Drink It')::TEXT AS who_should_drink,
      (data->>'$.Bottom Line')::TEXT AS bottom_line
    FROM
      read_json('{JSON_PATH}', format='auto') AS data;
""")

print("=" * 20)
print(f"Ingestion Complete")
print("=" * 20)
