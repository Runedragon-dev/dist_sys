import io
import json
import os
import threading
import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import ml
from PIL import Image
import base64
import torchvision.transforms as transforms
import torch
import os

import numpy as np
from PIL import Image
import pickle
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import inspect
import _operator
import torch_geometric
import typing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    torch.serialization.add_safe_globals([getattr])
    torch.serialization.add_safe_globals([GCNDepthEstimation])
    torch.serialization.add_safe_globals([set])
    torch.serialization.add_safe_globals([ImageToGraph])
    torch.serialization.add_safe_globals([GCNLayer])
    torch.serialization.add_safe_globals([GCNConv])
    torch.serialization.add_safe_globals([torch_geometric.nn.aggr.basic.SumAggregation])
    torch.serialization.add_safe_globals([torch_geometric.nn.dense.linear.Linear])
    torch.serialization.add_safe_globals([torch_geometric.inspector.Inspector])
    torch.serialization.add_safe_globals([torch_geometric.inspector.Signature])
    torch.serialization.add_safe_globals([torch_geometric.inspector.Parameter])
    torch.serialization.add_safe_globals([inspect._empty])
    torch.serialization.add_safe_globals([_operator.getitem])
    torch.serialization.add_safe_globals([typing.Union])
    torch.serialization.add_safe_globals([type])
    torch.serialization.add_safe_globals([int])
    torch.serialization.add_safe_globals([typing.OrderedDict])
    torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNDepthEstimation()
    model.load_state_dict(torch.load("GCN_model", weights_only=False)())
    model.to(device)
    return model

class ShapeNetDatasetFolders(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Проходим по каждой модели в корневой директории
        for model_folder in os.listdir(root_dir):
            model_path = os.path.join(root_dir, model_folder)
            model_file = os.path.join(model_path, "model.obj")

            # Проверяем, что путь содержит файл модели
            if os.path.exists(model_file):
                # Сохраняем пути к изображениям с разных ракурсов
                views = ["back.png", "front.png", "left.png", "right.png"]
                for view in views:
                    image_path = os.path.join(model_path, view)
                    if os.path.exists(image_path):
                        # Добавляем запись: каждый ракурс изображения + модель
                        self.data.append({
                            "model_file": model_file,
                            "image_file": image_path
                        })
def rgb_to_grayscale_3_channels(image: torch.Tensor) -> torch.Tensor:
    """
    Преобразует RGB-изображение в градации серого с сохранением 3 каналов.

    :param image: Тензор изображения с размерностью (C, H, W), где C=3.
    :return: Тензор изображения с градациями серого и размерностью (3, H, W).
    """
    if image.shape[0] != 3:
        raise ValueError("Ожидается изображение с 3 каналами (RGB).")

    # Коэффициенты для преобразования в оттенки серого
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
    grayscale = (weights.view(3, 1, 1) * image).sum(dim=0, keepdim=True)  # Размер (1, H, W)

    # Копируем серый канал в 3 канала
    grayscale_3_channels = grayscale.repeat(3, 1, 1)  # Размер (3, H, W)

    return grayscale_3_channels

def sample_points(points, num_samples):
    """
    Сэмплирует облако точек до фиксированного размера num_samples.

    :param points: Тензор размера (num_points, 3).
    :param num_samples: Число точек, до которого нужно уменьшить/увеличить облако.
    :return: Тензор размера (num_samples, 3).
    """
    num_points = points.shape[0]
    if num_points >= num_samples:
        # Случайное подвыборка
        indices = torch.randperm(num_points)[:num_samples]
        return points[indices]
    else:
        # Если точек меньше, дублируем их
        num_missing = num_samples - num_points
        duplicated_points = points[torch.randint(0, num_points, (num_missing,))]
        return torch.cat([points, duplicated_points])

def normalize_points_coords(points):
    """
    Нормализует облако точек.
    1. Вычитает центр масс.
    2. Нормализует расстояние точек от центра.

    :param points: Тензор размера (num_points, 3).
    :return: Нормализованный тензор.
    """
    # Вычисляем центр масс облака точек (среднее значение по каждой оси)
    # centroid = points.mean(dim=0)

    # # Сдвигаем облако точек, чтобы центр масс оказался в начале координат
    # points_centered = points - centroid

    # # Нормализуем расстояние от центра
    # # Нахождение максимального расстояния от центра
    # max_distance = torch.max(torch.norm(points_centered, dim=1))

    # # Нормализуем облако точек, деля на максимальное расстояние
    # points_normalized = points_centered / max_distance
    min_vals = torch.min(points, dim=0).values
    max_vals = torch.max(points, dim=0).values

    # Нормализация координат в диапазон [0, 1]
    points_normalized = (points - min_vals) / (max_vals - min_vals)
    return points_normalized*10

def normalize_point_cloud(points, num_samples = 1024):
    """
    Нормализует облако точек до фиксированного размера.

    :param points: Тензор размера (num_points, 3).
    :param num_samples: Число точек в выходном облаке.
    :return: Тензор размера (num_samples, 3).
    """
    num_points = points.shape[0]
    if num_points >= num_samples:
        return sample_points(points, num_samples)
    else:
        num_missing = num_samples - num_points
        duplicated_points = points[torch.randint(0, num_points, (num_missing,))]
        return torch.cat([points, duplicated_points])


class ZPointDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for model_image in [model for model in os.listdir(root_dir) if model.endswith(".png")]:


            if model_image==".ipynb_checkpoints":
              continue
            self.data.append({
                "model_file": root_dir + "/" + model_image[:-4]+".obj",
                "image_file": root_dir + "/" + model_image,
                "pickle_file": root_dir + "/" + model_image[:-4]+".pkl"
            })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      # Получаем путь к модели и изображению
      model_file = self.data[idx]["model_file"]
      image_file = self.data[idx]["image_file"]
      pickle_file = self.data[idx]["pickle_file"]

      # # Загрузка модели
      # mesh = load_objs_as_meshes([model_file], device="cuda",load_textures=False)
      # # Генерация облака точек с помощью sample_points_from_meshes\
      # sampled_points = sample_points_from_meshes(mesh,num_samples=1024)
      # point_cloud = sampled_points.squeeze(0)  # Получаем облако точек


      verts = []
      with open(pickle_file, 'rb') as file:
        verts = pickle.load(file)
      verts = torch.tensor(verts).reshape((1, 224, 224))
      if len(verts) == 0:
          raise ValueError(f"Model {model_file} contains no vertices.")


      # Загрузка и преобразование изображения
      image = Image.open(image_file).convert("RGB")

      if self.transform:
          image = self.transform(image)
      else:
          # Преобразование в тензор и нормализация
          image = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255.0
      image = rgb_to_grayscale_3_channels(image)
      return image, verts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_edges_8_neighbors(H, W):
    edges = []
    for i in range(H):
        for j in range(W):
            # Текущий индекс
            current_idx = i * W + j
            # Соседи (8 направлений)
            neighbors = [
                (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                (i, j - 1),                 (i, j + 1),
                (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
            ]
            for ni, nj in neighbors:
                if 0 <= ni < H and 0 <= nj < W:  # Проверяем, что сосед внутри изображения
                    neighbor_idx = ni * W + nj
                    edges.append((current_idx, neighbor_idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def generate_edges_24_neighbors(H, W):
    edges = []
    for i in range(H):
        for j in range(W):
            # Текущий индекс
            current_idx = i * W + j
            # Соседи (радиус 2)
            neighbors = [
                (i + di, j + dj)
                for di in range(-2, 3)  # -2, -1, 0, 1, 2
                for dj in range(-2, 3)  # -2, -1, 0, 1, 2
                if not (di == 0 and dj == 0)  # Исключаем самого себя
            ]
            for ni, nj in neighbors:
                if 0 <= ni < H and 0 <= nj < W:  # Проверяем, что сосед внутри изображения
                    neighbor_idx = ni * W + nj
                    edges.append((current_idx, neighbor_idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class ImageToGraph(nn.Module):
    def __init__(self, height, width, edges_num = 4):
        super(ImageToGraph, self).__init__()
        self.height = height
        self.width = width
        self.edges_num = edges_num

    def forward(self, x):
        # Размер изображения: [batch, 3, 224, 224]
        batch_size, channels, H, W = x.shape

        # Переводим изображение в форму [batch, 3, H*W]
        x_flat = x.view(batch_size, channels, -1).transpose(1, 2)  # [batch, H*W, 3]

        # Генерация графа (соседи)
        row, col = torch.meshgrid(torch.arange(H), torch.arange(W))  # [H, W] сетка
        edges = []
        if self.edges_num == 8:
          edges = generate_edges_8_neighbors(H, W)
        elif self.edges_num ==24:
          edges = generate_edges_24_neighbors(H, W)
        else:
          for i in range(H):
              for j in range(W):
                  if i < H - 1:  # добавляем нижнего соседа
                      edges.append((i * W + j, (i + 1) * W + j))
                  if j < W - 1:  # добавляем правого соседа
                      edges.append((i * W + j, i * W + (j + 1)))

          edges = torch.tensor(edges).t().contiguous()  # Получаем список рёбер (начало и конец)

        return x_flat, edges


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)


    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)

class GCNDepthEstimation_new(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=1 , edges_num = 4):
        super(GCNDepthEstimation_new, self).__init__()
        self.num_edges = edges_num
        self.image_to_graph = ImageToGraph(224, 224, edges_num=self.num_edges)

        self.gcn1 = GCNLayer(in_channels, hidden_channels)
        self.gcn2 = GCNLayer(hidden_channels, hidden_channels)
        self.gcn3 = GCNLayer(hidden_channels, out_channels)

        self.fc = nn.Linear(224 * 224 * out_channels, 224 * 224)

    def forward(self, x):
        batch_size, channels, H, W = x.shape
        x_flat, edge_index = self.image_to_graph(x)
        edge_index = edge_index.to(device)
        # Применяем GCN
        x = self.gcn1(x_flat, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.gcn3(x, edge_index)

        # Получаем итоговое предсказание
        x = x.view(batch_size, H, W, -1)  # Преобразуем обратно в изображение
        x = x.permute(0, 3, 1, 2)  # [batch, out_channels, H, W]

        return x

class GCNDepthEstimation(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=1):
        super(GCNDepthEstimation, self).__init__()
        self.image_to_graph = ImageToGraph(224, 224)

        self.gcn1 = GCNLayer(in_channels, hidden_channels)
        self.gcn2 = GCNLayer(hidden_channels, hidden_channels)
        self.gcn3 = GCNLayer(hidden_channels, out_channels)

        self.fc = nn.Linear(224 * 224 * out_channels, 224 * 224)

    def forward(self, x):
        batch_size, channels, H, W = x.shape
        x_flat, edge_index = self.image_to_graph(x)
        edge_index = edge_index.to(device)
        # Применяем GCN
        x = self.gcn1(x_flat, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.gcn3(x, edge_index)

        # Получаем итоговое предсказание
        x = x.view(batch_size, H, W, -1)  # Преобразуем обратно в изображение
        x = x.permute(0, 3, 1, 2)  # [batch, out_channels, H, W]

        return x


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TOKEN = ""
RESULT_API_URL = ""
#model = init()

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
    # model = GCNDepthEstimation()
    # model.load_state_dict(torch.load("/GCN_model")())
    #
    #обработка убрана для того чтобы не перегружать систему при тестах
    print("processing")
    return tensor

def send_result(id: int, result: torch.Tensor):
    # Преобразуем тензор в список для отправки JSON-данных
    with open("apilink.txt", 'r') as file:
        RESULT_API_URL = file.read()
    result_list = result.tolist()
    payload = json.dumps({"id": f"{str(id)}", "token": f"{str(TOKEN)}", "model": f"abc"},) #{str(result_list)}
    print(payload)
    print("sending result")
    try:
        response = requests.put(RESULT_API_URL, json=payload)
        print("response",response)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error sending result: {e}")

def process_and_send(id: int, photo_base64: str):
    global processing
    result = process_image(photo_base64)
    send_result(id, result)
    with lock:
        processing = False

@app.post("/upload")
async def upload_image(data: InputData, background_tasks: BackgroundTasks):
    """Обработчик API: принимает фото и запускает его обработку в фоне, если API свободен."""
    global processing
    with lock:
        if processing:
            return {"status": "busy"}
        processing = True

    background_tasks.add_task(process_and_send, data.id, data.photo)
    return {"status": "success"}

@app.get("/status")
def status():
    """Возвращает статус занятости API."""
    return {"busy": processing}
if __name__ == "__main__":

    port = 8081
    uvicorn.run(app, host="0.0.0.0", port=port)