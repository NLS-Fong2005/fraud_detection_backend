# <--- Imports --->
import os

class DocumentLoader:
    def __init__(self):
        self.GUIDELINES_DIRECTORY: str = os.path.abspath("./guidelines")
        

document_loader = DocumentLoader()