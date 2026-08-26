# Ambiente reproduzível para verificar o trabalho sem instalar nada na máquina.
#
# Escopo deliberado: CPU. A imagem roda os testes, confere os números do relatório contra
# results/ e faz inferência com os checkpoints de models/. NÃO treina — treino precisa de CUDA,
# de 1,9 GB de dataset e de horas de GPU, e uma imagem que prometesse isso seria uma imagem
# que ninguém testou. As versões de treino estão em requirements.txt e environment.md.
#
#   docker build -t antoniq-fruit .
#   docker run --rm antoniq-fruit                      # testes + conferência do relatório
#   docker run --rm -v "$PWD/results:/app/results" antoniq-fruit python scripts/08_report.py
#
#   # inferência numa imagem sua (monte a pasta que a contém)
#   docker run --rm -v "/caminho/das/imagens:/data" antoniq-fruit \
#          python scripts/run_inference.py --image /data/foto.png --no-image

FROM python:3.11-slim

# libgl1 e libglib2.0-0: o opencv-contrib-python (não-headless) linka contra elas.
# Manter o pacote não-headless porque é o que produziu os resultados — trocar por headless
# mudaria a dependência que o requirements.txt declara.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch de CPU vem antes: sem isto o ultralytics puxaria a build de CUDA (~2,5 GB) que a
# imagem não tem como usar.
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# O Ultralytics escreve config e baixa fontes no primeiro uso; sem apontar para /tmp ele tenta
# escrever em $HOME e avisa a cada execução.
ENV YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python -m pytest tests/ -q && python scripts/09_check_report.py"]
