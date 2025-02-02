import os
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import glob
from pytorch3d.io import load_obj
import pickle
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import inspect
import _operator
import torch_geometric
import typing



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ZPointDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Проходим по каждой модели в корневой директории
        for model_image in [model for model in os.listdir(root_dir) if model.endswith(".png")]:
            # Добавляем запись: каждый ракурс изображения + модель

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

model =  GCNDepthEstimation()
model.load_state_dict(torch.load("/content/GCN_model")())
