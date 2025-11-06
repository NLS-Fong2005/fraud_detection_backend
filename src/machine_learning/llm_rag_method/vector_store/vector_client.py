import chromadb

# <--- Configurations --->
class VectorClient:
    def __init__(self):
        self.vector_client = None
        
        if not (self.vector_client):
            self.__set_vector_connection__()
            print(f"Connection is established.\nHeartbeat: {self.vector_client.heartbeat()}") #type: ignore
        else:
            print(f"Connection already established! {self.vector_client.heartbeat()}")

    def __set_vector_connection__(self):
        try:
            self.vector_client = chromadb.HttpClient(host="localhost", port=8000)

            if (self.vector_client.heartbeat()):
                print("Connection Established.")
            else:
                print("Connection is not established")            
        except Exception as e:
            print(f"Something went wrong: {e}")

    def get_vector_connection(self):
        return self.vector_client

vector_client = VectorClient()

# Exclusive to Testing
if __name__ == "__main__":
    print(vector_client.get_vector_connection().heartbeat()) #type: ignore