import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime

# Função para obter dados históricos de Bitcoin da API
def obter_dados_bitcoin():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {'vs_currency': 'usd', 'days': '60', 'interval': 'daily'}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(by='date').reset_index(drop=True)
    return df

# Função de pré-processamento
def preprocess_data(df):
    # Adicionar features de data
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Selecionar as features e o target
    X = df[['close', 'day_of_week', 'month']]
    y = df['close']

    # Verificar dados para NaNs e imputar
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Normalizar os dados
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_y
