from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv
import os
from database import DB_Handler

logging.basicConfig(level=logging.INFO)

class Document_Tools:
    def __init__(self):
        load_dotenv()
        self.DOCS_PATH = os.getenv('DOCUMENTS_PATH')
    
    def _pdf_loader(self, data_path, files=None):
        pdf_document_loader = PyPDFDirectoryLoader(data_path)
        if files:
            doc = pdf_document_loader.load(files)
        else:
            doc = pdf_document_loader.load()
        return doc
    
    def _txt_loader(self, data_path, files=None):
        loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
        if files:
            doc = loader.load(files)
        else:
            doc = loader.load()
        return doc
    
    def _csv_loader(self, data_path, files=None):
        loader = DirectoryLoader(data_path, glob="**/*.csv", loader_cls=CSVLoader)
        if files:
            doc = loader.load(files)
        else:
            doc = loader.load()
        return doc
    
    def get_documents_list(self):
        docs = os.listdir(self.DOCS_PATH)
        return docs

    
    def load_documents(self, data_path) -> tuple[list[Document], list[str]]:
        logging.info(f"Loading documents from {data_path}...")
        files = os.listdir(data_path)

        pdf_files = [file for file in files if file.endswith('.pdf')]
        txt_files = [file for file in files if file.endswith('.txt')]
        csv_files = [file for file in files if file.endswith('.csv')]

        docs_names = pdf_files + txt_files + csv_files

        if pdf_files:
            doc_pdf = self._pdf_loader(data_path, pdf_files)
        if txt_files:
            doc_txt = self._txt_loader(data_path, txt_files)
        if csv_files:
            doc_csv = self._csv_loader(data_path, csv_files)

        docs = doc_pdf + doc_txt + doc_csv

        logging.info(f"Loaded {len(docs)} documents.")
        return docs, docs_names
    
    def load_single_document(self, file_path) -> Document:
        if file_path.endswith('.pdf'):
            doc = self._pdf_loader(file_path)
        elif file_path.endswith('.txt'):
            doc = self._txt_loader(file_path)
        elif file_path.endswith('.csv'):
            doc = self._csv_loader(file_path)
        else:
            logging.error(f"Unsupported file type: {file_path}")
            return None
        return doc

    def add_document(self, file):
        file_path = os.path.join(self.DOCS_PATH, file.name)

        with open(file_path, 'wb') as f:
            f.write(file.getvalue())

        db = Database()
        db.add(file.name)

    def remove_document(self, file):
        file_path = os.path.join(self.DOCS_PATH, file)
        print(file_path)
        os.remove(file_path)
        logging.info(f"Document {file} removed.")

        db = Database()
        db.remove(file)

    def clear_documents(self):
        files = os.listdir(self.DOCS_PATH)
        for file in files:
            file_path = os.path.join(self.DOCS_PATH, file)
            os.remove(file_path)
        
        db = Database()
        db.populate(reset=True)
    
    def split_documents(self, documents: list[Document], file_names: list[str]) -> dict[str, list]:
        logging.info(f"Splitting {len(documents)} documents...")

        chunks_dict = {}
        for doc, file_name in zip(documents, file_names):
            chunks = self.get_chunks(doc)
            chunks_dict[file_name] = chunks
 
        logging.info(f"Split into {sum(len(chunks) for chunks in chunks_dict.values())} chunks.")
        return chunks_dict
    
    def get_chunks(self, documents) -> list:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False
        )
        logging.info(f"Splitting {len(documents)} documents...")

        chunks = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(chunks)} chunks.")
        return chunks
    
class Database:
    def __init__(self): 
        load_dotenv()
        self.DOCS_PATH = os.getenv('DOCUMENTS_PATH')

    def populate(self, reset: bool = False):
        """
        if called will check the documents folder for new documents 
        and update the database.

        if reset is passed as True, it will completly wipe the
        database, and proceed to populate it.
        """
        db = DB_Handler()
        doct = Document_Tools()

        if reset:
            logging.info("Clearing Database...")
            logging.warn("Database is being cleared.")
            db.clear_database()

        docs, file_names = doct.load_documents(self.DOCS_PATH)
        chunks_dict = doct.split_documents(docs, file_names)
        logging.info("Populating the database...")
        db.add_to_chroma(chunks_dict)
    
    def add(self, filename):
        db = DB_Handler()
        doct = Document_Tools()
        file_path = os.path.join(self.DOCS_PATH, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found in the documents folder.")
        
        doc = doct.load_single_document(file_path)
        chunks = doct.get_chunks(doc)
        chunks_dict = {filename: chunks}

        logging.info("Adding document to the database...")
        db.add_to_chroma(chunks_dict)
        

    def remove(self, filename):
        db = DB_Handler()
        db.remove_document(filename)

