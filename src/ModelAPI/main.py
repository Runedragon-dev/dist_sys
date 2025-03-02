import io
import json
import threading
import ml
import torch
import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Any
import torchvision.transforms as transforms
from PIL import Image
import base64


TOKEN = ""
RESULT_API_URL = ""
model = ml.GCNDepthEstimation()
model.load_state_dict(torch.load("GCN_model")())

with open("token.txt", 'r') as file:
    TOKEN = file.read()
with open("apilink.txt", 'r') as file:
    RESULT_API_URL = file.read()

app = FastAPI()
processing = False
lock = threading.Lock()



class InputData(BaseModel):
    id: int
    photo: bytes

def process_image(base64_str: bytes) -> str:
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Конвертируем в RGB


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(image)

    return tensor


def send_result(id: int, result: str):
    """Отправка результата в API"""
    payload = {"id": id, "token": TOKEN, "model": result}
    try:
        response = requests.post(RESULT_API_URL, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error sending result: {e}")

@app.post("/upload")
async def upload_image(data: InputData, background_tasks: BackgroundTasks):
    """Обработчик API: принимает данные, сразу отвечает, обрабатывает в фоне"""
    background_tasks.add_task(process_and_send, data.id, data.photo)
    return {"status": "success"}

def process_and_send(id: int, photo_bytes: bytes):
    """Фоновая обработка фото и отправка результата"""
    result = process_image(photo_bytes)
    send_result(id, result)

@app.get("/status")
def status():
    """Возвращает статус занятости API"""
    return {"busy": processing}

@app.post("/upload")
async def upload_image(data: InputData, background_tasks: BackgroundTasks):
    """Получает фото и запускает обработку в фоне"""
    global processing
    with lock:
        if processing:
            return {"status": "busy"}
        processing = True

    background_tasks.add_task(process_and_send, data.id, data.photo)
    return {"status": "success"}

def process_and_send(id: int, photo_bytes: bytes):
    """Фоновая обработка фото"""
    global processing
    result = process_image(photo_bytes)
    send_result(id, result)
    with lock:
        processing = False


if __name__ == "__main__":
    import os

    port = 8081
    uvicorn.run(app, host="0.0.0.0", port=port)
