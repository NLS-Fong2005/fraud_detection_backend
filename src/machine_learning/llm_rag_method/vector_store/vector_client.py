import weaviate
from typing import Optional

# <--- Configurations --->
class VectorClient:
    def __init__(self):
        self.vector_client: Optional[weaviate.client.WeaviateClient] = None
        
        if not (self.vector_client):
            self.__set_vector_connection__()
        else:
            print("Already connected to Weaviate.")

    def __set_vector_connection__(self) -> None:
        try:
            if self.vector_client and self.vector_client.is_connected():
                print("Vector client is already connected.")
                return None
            else:
                self.vector_client = weaviate.connect_to_local() #type: ignore
                print("Successfully connected to Weaviate.")
        except Exception as e:
            print(f"Something went wrong: {e}")
            exit()

    def get_vector_connection(self) -> weaviate.client.WeaviateClient:
        if not self.vector_client:
            self.__set_vector_connection__()
        return self.vector_client #type: ignore 
    
    def close_vector_connection(self) -> None:
        if (self.vector_client and self.vector_client.is_connected()):
            self.vector_client.close()
            return None
        print("Vector Client is already closed!")
        return None

vector_client = VectorClient()

# Exclusive to Testing
if __name__ == "__main__":
    print(vector_client.get_vector_connection().is_connected())
    vector_client.close_vector_connection()