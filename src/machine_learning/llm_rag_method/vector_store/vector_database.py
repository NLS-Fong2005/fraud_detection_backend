# <--- Imports --->
import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.config import (
    Configure,
    Property,
    DataType
)
from src.machine_learning.llm_rag_method.vector_store.vector_client import vector_client

# <--- Configurations --->
COLLECTION_NAME: str = "guidelines_collection"

class VectorCollection:
    def __init__(self):
        self.collection = None
        self.vector_client = vector_client.get_vector_connection()

    def __create_vector_collection__(self):
        if self.vector_client.collections.exists(name=COLLECTION_NAME):
            return f"Collection '{COLLECTION_NAME}' already exists!"
        try:
            self.vector_client.collections.create(
                name=COLLECTION_NAME,
                properties=[
                    Property(name="header", data_type=DataType.TEXT),
                    Property(name="info", data_type=DataType.TEXT)
                ],
                vectorizer_config=[
                    Configure.NamedVectors.text2vec_openai(
                        name='chatgpt',
                        source_properties=['info'],
                        model='ada',
                        model_version='002'
                    )
                ]
            )
        except Exception as e:
            print(f"Something went wrong: {e}")

    def get_vector_collection(self) -> weaviate.collections.Collection:
        self.collection = self.vector_client.collections.get(name=COLLECTION_NAME)
        return self.collection

    def __fetch_all_objects__(self) -> list:
        try:
            objects = self.get_vector_collection().query.fetch_objects(limit=400).objects
            documents = []

            for obj in objects:
                documents.append(
                    {
                        "UUID" : str(obj.uuid),
                        "Properties" : obj.properties,
                        "Vector" : obj.vector
                    }
                )

            print(f"📖 Retrieved {len(documents)} documents from collection: {COLLECTION_NAME}")
            return documents
        except Exception as e:
            print(f"Error retrieving documents from collection: {e}")
            return []
        
    def fetch_object_from_header(self, header: str) -> list:
        try:
            context = []
            documents = self.__fetch_all_objects__()

            for document in documents:
                if header in document["Properties"]["Header"]:
                    context.append(document)

            print(f"📖 Retrieved {len(context)} documents from collection: {COLLECTION_NAME}")
            return context
        except Exception as e:
            print(f"Something went wrong: {e}")
            return []

# Testing Purposes
if __name__ == "__main__":
    pass