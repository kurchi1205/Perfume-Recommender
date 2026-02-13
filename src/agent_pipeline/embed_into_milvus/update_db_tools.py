# check if the output collection or db exists
# if it is not there then create it
# read through the jsonl
# take each emtry and check if it is in the collection, if not embed reqired items and push it to the collection


import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, "../../")

from langchain_core.tools import tool
from langchain.agents import create_agent
from milvus_setup.create_db import create_connection, create_db
from pymilvus import db
from pymilvus import MilvusClient, DataType
from langchain_ollama import ChatOllama
from utils import build_record, init_bge_embedder



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

perfumes_list = []
embedded_urls = set()
unembedded_urls = set()
DB_NAME = "perfume_db"
COLLECTIONS_NAME = "perfume_collection"

@tool
def check_if_db_exists() -> bool:
    "Check if the db exists. Returns True if it exists, False otherwise."
    logger.info("Checking if database '%s' exists...", DB_NAME)
    create_connection()
    existing_dbs = db.list_database()

    if DB_NAME not in existing_dbs:
        logger.info("Database '%s' not found. Existing dbs: %s", DB_NAME, existing_dbs)
        return False
    logger.info("Database '%s' exists.", DB_NAME)
    return True

@tool
def create_milvus_db(db_name: str) -> str:
    "Create a new Milvus database. Only call this if check_if_db_exists returned False."
    logger.info("Creating database '%s'...", db_name)
    create_db()
    logger.info("Database '%s' created.", db_name)
    return "Database created"

def init_milvus_client(db_name):
    "Initialize the Milvus client"
    logger.info("Initializing Milvus client for database '%s'...", db_name)
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )
    client.using_database(db_name)
    logger.info("Milvus client ready (db='%s').", db_name)
    return client

@tool
def check_if_collection_exists(db_name: str, collection_name: str) -> bool:
    "Check if a collection exists in the database. Returns True if it exists, False otherwise."
    logger.info("Checking if collection '%s' exists in db '%s'...", collection_name, db_name)
    global client
    client = init_milvus_client(db_name)
    if client.has_collection(collection_name):
        logger.info("Collection '%s' exists.", collection_name)
        return True
    logger.info("Collection '%s' not found.", collection_name)
    return False


def create_schema_for_collection():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=36)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="gender", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="top_notes", datatype=DataType.VARCHAR, max_length=2000, is_array=True)
    schema.add_field(field_name="middle_notes", datatype=DataType.VARCHAR, max_length=2000, is_array=True)
    schema.add_field(field_name="base_notes", datatype=DataType.VARCHAR, max_length=2000, is_array=True)
    schema.add_field(field_name="main_accords", datatype=DataType.VARCHAR, max_length=2000, is_array=True)
    schema.add_field(field_name="moods_embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
    return schema


@tool
def create_collection(db_name: str, collection_name: str) -> str:
    "Create a new Milvus collection. Only call this if check_if_collection_exists returned False."
    logger.info("Creating collection '%s'...", collection_name)
    global client
    client = init_milvus_client(db_name)
    schema = create_schema_for_collection()
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="moods_embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="vector_index",
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )
    logger.info("Collection '%s' created.", collection_name)
    return "Collection created"


embedder = init_bge_embedder()
@tool
def insert_into_collection(collection_name: str, input_path: str) -> str:
    "Insert perfume data from a JSONL file into a Milvus collection."
    logger.info("Starting insertion into collection '%s' from '%s'...", collection_name, input_path)
    global client
    batch = []
    total = 0
    BATCH_SIZE = 10
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            record = build_record(embedder, item)

            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                client.insert(collection_name=collection_name, data=batch)
                total += len(batch)
                logger.info("Inserted batch — %d records so far.", total)
                batch.clear()

    # final flush
    if batch:
        client.insert(collection_name=collection_name, data=batch)
        total += len(batch)

    logger.info("Insertion complete. Total records inserted: %d", total)
    return "Insertion complete"



# SYSTEM_PROMPT = (
#         "You are a database engineer. You MUST call all the tools to complete tasks — "
#         "Must follow the instructions.\n"
#         "Start by calling check_if_db_exists with db_name and collection_name. If it returns False, call create_milvus_db.\n"
#         "Then proceed with check_if_collection_exists, if it returns False, call create_collection.\n"
#         # "Then call insert_into_collection with Collection_name and INPUT_PATH as parameter to insert data.\n"
#     )


# def build_agent():
#     """Build the ReAct agent with llama3.1:8b and the 5 tools."""
#     model = ChatOllama(model="llama3.1:8b", temperature=0)
#     INPUT_PATH = Path("../../../datasets/perfumes_with_moods.jsonl")
#     agent = create_agent(
#         model=model,
#         tools=[check_if_db_exists, create_milvus_db, check_if_collection_exists, create_collection, insert_into_collection],
#         system_prompt=SYSTEM_PROMPT + f"\n\n INPUT_PATH: {INPUT_PATH}, Database_name: {DB_NAME}, Collection_name: {COLLECTIONS_NAME}",
#     )
#     return agent


# if __name__=="__main__":
#     agent = build_agent()
#     INPUT_PATH = Path("../../../datasets/perfumes_with_moods.jsonl")
#     result = agent.invoke(
#         {"messages": [{"role": "user", "content": (
#             f"Set up the database '{DB_NAME}' and collection '{COLLECTIONS_NAME}', "
#             f"then insert data from {INPUT_PATH} into it."
#         )}]}
#     )

#     # Print final agent message
#     final_message = result["messages"][-1]
#     print(f"\nAgent: {final_message.content}")
