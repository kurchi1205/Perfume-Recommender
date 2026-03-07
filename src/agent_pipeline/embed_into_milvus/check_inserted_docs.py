from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# Switch to your database
client.using_database("perfume_db")

collections = client.list_collections()
print(f"Collections in 'perfume_db': {collections}")

# client.drop_collection(
#     collection_name="perfume_collection"
# )
client.load_collection(
    collection_name="perfume_collection"
)

print(client.get_load_state(
        collection_name="perfume_collection"
    ))
# Query first 10 entities
results = client.query(
    collection_name="perfume_collection",
    filter="",  # Empty filter returns all
    output_fields=["*"],  # Return all fields
    limit=10
)

# Print results
for i, entity in enumerate(results, 1):
    print(f"\n--- Entity {i} ---")
    for key, value in entity.items():
        print(f"{key}: {value}")