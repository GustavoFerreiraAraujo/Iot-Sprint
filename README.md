Projeto: Previsão de Preço de Bitcoin com Flask e LSTM

Descrição:

Este projeto utiliza a biblioteca Keras do TensorFlow para construir e treinar um modelo LSTM (Long Short-Term Memory) para prever o preço futuro do Bitcoin. A aplicação Flask fornece uma API para acessar as previsões e os picos de alta e baixa estimados.

Tecnologias Utilizadas:

Python
Pandas
NumPy
TensorFlow (Keras)
Flask
Requests
Instalação:

Crie um ambiente virtual e instale as dependências:

Bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Uso:

Execute o aplicativo Flask:

Bash
python app.py
Use o código com cuidado.

Acesse a API para obter os picos de alta e baixa previstos para os próximos 7 dias:

Bash
curl http://127.0.0.1:5000/pico-alta-baixa

A resposta será um JSON contendo a data, o preço previsto, o pico alto e o pico baixo para cada dia futuro.

Como Funciona:

O aplicativo obtém dados históricos de preços do Bitcoin da API do CoinGecko.
Os dados são pré-processados, incluindo normalização e criação de janelas de dados para o modelo LSTM.
O modelo LSTM é treinado para aprender o padrão temporal dos preços do Bitcoin.
A API permite que você obtenha os preços previstos para os próximos 7 dias, juntamente com estimativas de picos altos e baixos com base em uma amplitude definida.
