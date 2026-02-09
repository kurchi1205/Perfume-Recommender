from pymilvus import connections, utility
from pymilvus import db

def create_connection():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )

    print("✅ Connected to Milvus")

def create_db():
    DB_NAME = "perfume_db"
    existing_dbs = db.list_database()

    if DB_NAME not in existing_dbs:
        db.create_database(DB_NAME)
        print(f"✅ Database created: {DB_NAME}")
    else:
        print(f"ℹ️ Database already exists: {DB_NAME}")

    db.using_database(DB_NAME)




if __name__ == "__main__":
    create_connection()
    create_db()
