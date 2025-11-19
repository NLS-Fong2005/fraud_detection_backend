from pydantic import BaseModel

class DataObject(BaseModel):
    message_content: str
    sent_time: str
    source_ip: str
    source_location: str