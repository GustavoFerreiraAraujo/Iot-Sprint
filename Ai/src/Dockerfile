# Usando a imagem base do Python 3.8
FROM python:3.8-slim

# Definindo o diretório de trabalho
WORKDIR /app

# Copiando o arquivo de requisitos (requirements.txt) para o diretório de trabalho
COPY requirements.txt .

# Instalando as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiando o restante do código da aplicação para o diretório de trabalho
COPY . .

# Expondo a porta 5050 que o Flask usará
EXPOSE 5050

# Comando para executar a aplicação Flask
CMD ["python", "api.py"]
