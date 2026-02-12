# check if the output collection or db exists
# if it is not there then create it
# read through the jsonl
# take each emtry and check if it is in the collection, if not embed reqired items and push it to the collection


import glob
from http import client
import json
import logging
from pathlib import Path

from langchain_core.tools import tool
from langchain.agents import create_agent
from milvus_setup.create_db import create_connection, create_db
from pymilvus import db
from pymilvus import MilvusClient, DataType
from langchain_ollama import ChatOllama



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

perfumes_list = []
embedded_urls = set()
unembedded_urls = set()
DB_NAME = "perfume_db"


@tool
def check_if_db_exists():
    "Check if the db exists"
    create_connection()
    existing_dbs = db.list_database()

    if DB_NAME not in existing_dbs:
        return False
    return True

@tool
def create_db():
    "Create a new Milvus database if it doesn't exist"
    create_db()

def init_milvus_client():
    "Initialize the Milvus client"
    global client
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )

@tool
def check_if_collection_exists(collection_name):
    "Check if the collection exists"
    global client
    init_milvus_client()
    if client.has_collection(collection_name):
        return True
    return False


def create_schema_for_collection():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=10)
    schema.add_field(field_name="name", datatype=DataType.FLOAT_VECTOR, dim=1536)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="gender", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="top_notes", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="middle_notes", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="base_notes", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="main_accords", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="moods", datatype=DataType.VARCHAR, max_length=512)
    return schema


@tool
def create_collection(collection_name):
    "Create a new Milvus collection"
    global client
    schema = create_schema_for_collection()
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    


@tool
def embed_perfumes():
    "Embed perfumes into Milvus"
    global client



SYSTEM_PROMPT = (
        "You are a software engineer. Your job is to "
        "1. Check if the perfume db exists. "
        "2. If it doesn't exist, create it. "
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
        tools=[check_if_db_exists, create_db, check_if_collection_exists, create_collection, insert_into_collection],
        prompt=SYSTEM_PROMPT + f"\n\n Input file: {INPUT_PATH}",
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
