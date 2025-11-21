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

app = FastAPI()

@app.get("/health_check")
async def health_check():
    return {"status": "Healthy"}

def process_csv(file_object):
    try:
        csv_chunks = pd.read_csv(file_object, chunksize=10)

        for chunk in csv_chunks:
            chunk.dropna(inplace=True)

            for _, row in chunk.iterrows():
                row_dict = row.to_dict()
                row_json = json.dumps(row_dict)

                yield f"{row_json}\n"
    except Exception as e:
        error_detail = json.dumps({"error": f"Stream Interrupted: {e}"})
        yield f"{error_detail}\n"

@app.post("/upload/stream-csv")
def upload_and_stream_csv(file: UploadFile = File(...)):
    return StreamingResponse(
        process_csv(file.file), #type: ignore
        media_type="application/x-ndjson"
    )