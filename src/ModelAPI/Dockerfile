FROM continuumio/anaconda3

WORKDIR /app

COPY requirements.txt .
COPY token.txt .
COPY apilink.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
