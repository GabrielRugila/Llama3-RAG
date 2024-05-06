import re
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3:8b")
    return embeddings

class DB_Handler:
    def __init__(self):
        load_dotenv()
        self.CHROMA_PATH = os.getenv('CHROMA_PATH')
    
    def get_db(self):
        db = Chroma(
            persist_directory=self.CHROMA_PATH, 
            embedding_function=get_embedding_function()
        )
        return db
    
    def get_db_with_collection(self, collection_name="default"):
        db = Chroma(
            persist_directory=self.CHROMA_PATH, 
            embedding_function=get_embedding_function(),
            collection_name=collection_name
        )
        return db
    
    def get_list_of_collections(self):
        db = self.get_db()
        collections = db.list_collections()
        return collections
    
    def get_collections(self, collections: list[str]):
        db = self.get_db()
        collections = db.get(include=collections)
        return collections
    
    def collection_naming(self, file_name):
        name = file_name.lower()
        if not re.fullmatch(r'^[a-z0-9][a-z0-9._-]*[a-z0-9]$', name):
            raise ValueError("Invalid collection name. \nReceived: {name}\nPlease use only lowercase letters, numbers, dots, underscores, and hyphens. \nName must start and end with a letter or number.")
        name = re.sub(r'\.+', '.', name)
        return name

    def add_to_chroma(self, chunks_dict: dict[str, list]):
        for file_name, chunks in chunks_dict.items():
            file_name = self.collection_naming(file_name)
            db = self.get_db_with_collection(collection_name=file_name)
            chunks_with_ids = self.chunk_ids(chunks)

            existing_items = db.get(include=[])  
            existing_ids = set(existing_items["ids"])
            logging.info(f"Documents in DB: {len(existing_ids)}")

            new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

            if new_chunks:
                logging.info(f"New Documents Found: {len(new_chunks)}")
                for chunk in tqdm(new_chunks, desc="Adding documents"):
                    db.add_documents([chunk], ids=[chunk.metadata["id"]])
                db.persist()
            else:
                logging.info("DB is up-to-date")

    def clear_database(self):
        db = self.get_db()
        db.delete_collection()
        logging.info("Database cleared.")

    def chunk_ids(self, chunks):
        last_page_id = None
        chunk_idx = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            page_id = f"{source}:{page}"

            if page_id == last_page_id:
                chunk_idx += 1
            else:
                chunk_idx = 0

            chunk_id = f"{page_id}:{chunk_idx}"
            last_page_id = page_id

            chunk.metadata["id"] = chunk_id

        return chunks

    def remove_document(self, document_name):
        db = self.get_db_with_collection(collection_name=document_name)
        db.delete_collection()
        logging.info(f"Collection {document_name} removed from database.")
                

