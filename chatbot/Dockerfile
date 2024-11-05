# Usar uma imagem base do Ubuntu
FROM ubuntu:20.04

# Atualizar o sistema e instalar Python, pip e outras dependências necessárias
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho
WORKDIR /app

# Copiar o arquivo requirements.txt para a imagem
COPY requirements.txt .

# Instalar as dependências
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação para a imagem
COPY api.py .

# Expor a porta que a aplicação irá rodar
EXPOSE 5000

# Comando para executar a aplicação
CMD ["python3", "api.py"]
