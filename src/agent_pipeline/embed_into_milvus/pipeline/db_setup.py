"""
Stage 4 — DB Setup
Idempotent: creates perfume_db and perfume_collection if they don't exist.
No LLM involved — pure deterministic setup.
"""
import logging

from pymilvus import (
    MilvusClient,
    DataType,
    connections,
    db,
)

logger = logging.getLogger(__name__)

MILVUS_URI   = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
DB_NAME      = "perfume_db"
COLLECTION   = "perfume_collection"
VECTOR_DIM   = 1024


def get_client() -> MilvusClient:
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    client.using_database(DB_NAME)
    return client


def ensure_db() -> None:
    connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
    existing = db.list_database()
    if DB_NAME not in existing:
        db.create_database(DB_NAME)
        logger.info("Created database '%s'", DB_NAME)
    else:
        logger.info("Database '%s' already exists", DB_NAME)


def ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION):
        logger.info("Collection '%s' already exists", COLLECTION)
        return

    schema = client.create_schema(enable_dynamic_field=True)

    schema.add_field("id",           DataType.VARCHAR, max_length=36,    is_primary=True, auto_id=False)
    schema.add_field("name",         DataType.VARCHAR, max_length=500)
    schema.add_field("description",  DataType.VARCHAR, max_length=65535)
    schema.add_field("url",          DataType.VARCHAR, max_length=65535)
    schema.add_field("brand",        DataType.VARCHAR, max_length=200)
    schema.add_field("gender",       DataType.VARCHAR, max_length=50)
    schema.add_field("top_notes",    DataType.VARCHAR, max_length=2000)
    schema.add_field("middle_notes", DataType.VARCHAR, max_length=2000)
    schema.add_field("base_notes",   DataType.VARCHAR, max_length=2000)
    schema.add_field("main_accords", DataType.VARCHAR, max_length=2000)
    # New fields
    schema.add_field("moods",        DataType.VARCHAR, max_length=2000)
    schema.add_field("summary",      DataType.VARCHAR, max_length=65535)
    schema.add_field("moods_embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="moods_embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        index_name="vector_index",
        params={"nlist": 128},
    )

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection '%s'", COLLECTION)


def setup() -> MilvusClient:
    """Run full idempotent setup and return a ready client."""
    ensure_db()
    client = get_client()
    ensure_collection(client)
    return client
