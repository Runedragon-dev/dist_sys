import pytest
from fastapi.testclient import TestClient
from main import app, processing, process_image, send_result, process_and_send
import main
import torch
import requests
from PIL import Image
import io
import base64
import json
import os

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def setup_files(tmpdir):
    # Создаем временные файлы token.txt и apilink.txt
    token_file = tmpdir.join("token.txt")
    token_file.write("test_token")

    api_file = tmpdir.join("apilink.txt")
    api_file.write("http://test-api.com")

    # Меняем текущую директорию на временную
    original_dir = os.getcwd()
    os.chdir(tmpdir)
    yield
    os.chdir(original_dir)

@pytest.fixture(autouse=True)
def reset_processing():
    # Сбрасываем processing перед каждым тестом
    with main.lock:
        main.processing = False

def test_status_not_busy(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"busy": False}

def test_status_busy(client):
    main.processing = True
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"busy": True}

def test_upload_busy(client):
    main.processing = True
    response = client.post("/upload", json={"id": 1, "photo": "base64_encoded_str"})
    assert response.status_code == 200
    assert response.json() == {"status": "busy"}

def test_upload_success(client):
    assert main.processing is False
    response = client.post("/upload", json={"id": 1, "photo": "base64_encoded_str"})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert main.processing is True

def test_process_image():
    # Создаем тестовое изображение
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    tensor = process_image(img_base64.encode('utf-8'))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 100, 100)

def test_send_result(client, mocker):
    mock_put = mocker.patch('requests.put')
    test_tensor = torch.tensor([1.0, 2.0])

    send_result(123, test_tensor)

    expected_url = "http://test-api.com"
    expected_payload = json.dumps({
        "id": "123",
        "token": "test_token",
        "model": "abc"
    }, separators=(',', ':'))  # Учитываем компактный вывод json.dumps
    mock_put.assert_called_once_with(expected_url, json=expected_payload)

def test_send_result_exception(client, mocker, caplog):
    mock_put = mocker.patch('requests.put')
    mock_put.side_effect = requests.exceptions.ConnectionError("Connection error")

    send_result(123, torch.tensor([1.0]))

    assert "Error sending result: Connection error" in caplog.text

def test_process_and_send(mocker):
    mock_process = mocker.patch('main.process_image', return_value=torch.tensor([1.0]))
    mock_send = mocker.patch('main.send_result')

    main.processing = True

    process_and_send(123, "base64_str")

    mock_process.assert_called_once_with("base64_str")
    mock_send.assert_called_once_with(123, torch.tensor([1.0]))
    assert main.processing is False