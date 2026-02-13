# check if the output collection or db exists
# if it is not there then create it
# read through the jsonl
# take each emtry and check if it is in the collection, if not embed reqired items and push it to the collection


import json
import logging
from pathlib import Path
from pydoc import cli
import sys
sys.path.insert(0, "../../")
from langchain_core.tools import tool
from langchain.agents import create_agent
from milvus_setup.create_db import create_connection, create_db
from pymilvus import db
from pymilvus import MilvusClient, DataType
from langchain_ollama import ChatOllama
from utils import build_record



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
def check_if_db_exists():
    "Check if the db exists"
    logger.info("Checking if database '%s' exists...", DB_NAME)
    create_connection()
    existing_dbs = db.list_database()

    if DB_NAME not in existing_dbs:
        logger.info("Database '%s' not found. Existing dbs: %s", DB_NAME, existing_dbs)
        return False
    logger.info("Database '%s' exists.", DB_NAME)
    return True

@tool
def create_milvus_db():
    "Create a new Milvus database if it doesn't exist"
    logger.info("Creating database '%s'...", DB_NAME)
    create_db()
    logger.info("Database '%s' created.", DB_NAME)

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
def check_if_collection_exists(db_name, collection_name):
    "Check if the collection exists"
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
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=10)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, dim=1536)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="gender", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="top_notes", datatype=DataType.VARCHAR, max_length=100, is_array=True)
    schema.add_field(field_name="middle_notes", datatype=DataType.VARCHAR, max_length=100, is_array=True)
    schema.add_field(field_name="base_notes", datatype=DataType.VARCHAR, max_length=100, is_array=True)
    schema.add_field(field_name="main_accords", datatype=DataType.VARCHAR, max_length=100, is_array=True)
    schema.add_field(field_name="moods_embedding", datatype=DataType.FLOAT_VECTOR, max_length=512)
    return schema


@tool
def create_collection(collection_name):
    "Create a new Milvus collection"
    logger.info("Creating collection '%s'...", collection_name)
    global client
    schema = create_schema_for_collection()
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    logger.info("Collection '%s' created.", collection_name)
    

@tool
def insert_into_collection(collection_name, path: str):
    "Insert perfume data into a Milvus collection"
    logger.info("Starting insertion into collection '%s' from '%s'...", collection_name, path)
    global client
    batch = []
    total = 0
    BATCH_SIZE = 100
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            record = build_record(item)

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



SYSTEM_PROMPT = (
        "You are a software engineer. Your job is to "
        "1. Check if the perfume db exists. "
        "2. If the tool returns false, create the db. "
        "3. If it does exist, check if the perfume collection exists."
        "4. If it doesn't exist, create it. "
        "5. Read the input jsonl file for each record do the following: "
        "6. Form Embedding for the 5 sensory moods. "
        "7. Insert the perfume data into the perfume collection."
    )


def build_agent():
    """Build the ReAct agent with llama3.1:8b and the 3 tools."""
    model = ChatOllama(model="llama3.1:8b", temperature=0)
    INPUT_PATH = Path("../../../datasets/perfumes_with_moods.jsonl")
    agent = create_agent(
        model=model,
        tools=[check_if_db_exists, create_milvus_db, check_if_collection_exists, create_collection, insert_into_collection],
        system_prompt=SYSTEM_PROMPT + f"\n\n Input file: {INPUT_PATH}, Database_name: {DB_NAME}, Collection_name: {COLLECTIONS_NAME}",
    )
    return agent


if __name__=="__main__":
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Insert the perfume entries in the collection."}]}
    )

    # Print final agent message
    final_message = result["messages"][-1]
    print(f"\nAgent: {final_message.content}")
