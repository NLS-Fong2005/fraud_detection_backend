# <--- Imports --->
import io
import pandas as pd
import json
from fastapi import (
    FastAPI,
    UploadFile,
    File,
)
from fastapi.responses import StreamingResponse

from src.machine_learning.llm_rag_method.models.object_model import DataObject

app = FastAPI()

@app.get("/health_check")
async def health_check():
    return {"status": "Healthy"}

def is_column_in(dataframe: pd.DataFrame) -> bool:
    if "Message" not in dataframe.columns:
        return False
    
    if "Sent_Time" not in dataframe.columns:
        return False
    
    if "Source_IP" not in dataframe.columns:
        return False

    if "Source_Location" not in dataframe.columns:
        return False
    
    return True

def process_csv(file_object):
    try:
        csv_chunks = pd.read_csv(file_object, chunksize=10)

        first_chunk = True


        for chunk in csv_chunks:
            if first_chunk:
                if not is_column_in(dataframe=chunk):
                    error_detail = json.dumps(
                        {
                            "error": f"Invalid CSV Format: Missing required columns."
                        }
                    )

                    yield f"{error_detail}\n"
                    return

                first_chunk = False

            chunk.dropna(inplace=True)

            for _, row in chunk.iterrows():
                row_dict = row.to_dict()
                
                row_object: DataObject = DataObject(
                    message_content=row_dict["Message"],
                    sent_time=row_dict["Sent_Time"],
                    source_ip=row_dict["Source_IP"],
                    source_location=row_dict["Source_Location"]
                )

                row_json = row_object.model_dump_json()

                yield f"{row_json}\n"
    except Exception as e:
        error_detail = json.dumps({"error": f"Stream Interrupted: {e}"})
        yield f"{error_detail}\n"

@app.post("/upload/stream-csv")
async def upload_and_stream_csv(file: UploadFile = File(...)):
    content_bytes = await file.read()

    csv_buffer = io.StringIO(content_bytes.decode("utf-8"))
    return StreamingResponse(
        process_csv(csv_buffer), #type: ignore
        media_type="application/x-ndjson"
    )