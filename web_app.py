import json

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import os
import zipfile
import time
import torch
from datetime import timedelta
from marker.convert import convert_single_pdf
from marker.models import load_all_models

app = FastAPI()
model_lst = None


@app.get("/health")
def health():
    if torch.cuda.is_available():
        return {
            "status": "UP",
            "device": "GPU",
            "available_GPUs": torch.cuda.device_count(),
            "current_GPU": torch.cuda.current_device(),
            "current_GPU_name": torch.cuda.get_device_name(torch.cuda.current_device())
        }
    else:
        return {
            "status": "UP",
            "device": "CPU"
        }


@app.post("/parse", timeout=timedelta(seconds=300))
async def parse(upload_file: UploadFile = File(...),
                max_pages: int = None,
                start_page: int = None,
                ocr_all_pages: bool = False,
                langs: str = None,
                batch_multiplier: int = 2
                ):
    upload_file_content = await upload_file.read()
    upload_file_name, extension = os.path.splitext(upload_file.filename)
    if extension != '.pdf':
        return {"error": "Please upload a PDF file"}

    langs = langs.split(",") if langs else None

    start = time.time()
    full_text, images, out_meta = convert_single_pdf(upload_file_content, model_lst, max_pages=max_pages,
                                                     langs=langs,
                                                     batch_multiplier=batch_multiplier, start_page=start_page,
                                                     ocr_all_pages=ocr_all_pages)
    print(f"UploadFile {upload_file.filename} parse used time: {time.time() - start}")
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(upload_file_name + ".md", full_text)
        zip_file.writestr(upload_file_name + "_meta.json", json.dumps(out_meta))
        for filename, image in images.items():
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            zip_file.writestr(filename, image_bytes.read())
    # 将 ZIP 文件的指针移到开始位置
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={upload_file_name}.zip"}
    )


if __name__ == "__main__":
    model_lst = load_all_models()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=16586)
