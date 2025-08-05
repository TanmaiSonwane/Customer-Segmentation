import pandas as pd
from pymongo import MongoClient

# 1. Load your CSV file
df = pd.read_csv("notebooks/data/clustered_data.csv")  # Make sure this CSV is present

# 2. Connect to your MongoDB Atlas cluster
client = MongoClient("Mongo_db_key")
# 3. Select the database and collection
db = client["ineuron"]  # Replace with your DB name
collection = db["customer_segmentation"]         # Replace with your collection name

# Optional: Clear old data to avoid duplicates
collection.delete_many({})

# Convert DataFrame to dictionary format and insert
data_dict = df.to_dict("records")
collection.insert_many(data_dict)

print(f"Inserted {len(data_dict)} records into MongoDB.")
print(collection.count_documents({}))
