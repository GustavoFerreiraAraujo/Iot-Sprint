import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, jsonify
from datetime import datetime, timedelta
import requests

# Inicializando o Flask
app = Flask(__name__)

# Função para obter dados históricos de Bitcoin da API
def obter_dados_bitcoin():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '60',  # Últimos 60 dias para ter mais dados
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(by='date').reset_index(drop=True)
    return df

# Carregar e preparar os dados
data = obter_dados_bitcoin()
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month

X = data[['close', 'day_of_week', 'month']]
y = data['close']

X = X.fillna(X.mean())
y = y.fillna(y.mean())

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

def criar_janelas(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 7
X_lstm, y_lstm = criar_janelas(X_scaled, look_back)
y_lstm = y_lstm[:len(X_lstm)]

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Construir o modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Função para prever o futuro
def forecast_future(model, last_data, look_back, n_future):
    future_predictions = []
    current_step = last_data.reshape(1, look_back, X_train.shape[2])
    for _ in range(n_future):
        pred = model.predict(current_step)
        future_predictions.append(pred[0, 0])
        current_step = np.roll(current_step, -1, axis=1)
        current_step[0, -1, 0] = pred[0, 0]
    return np.array(future_predictions)

# Simular picos de alta e baixa
def calcular_picos(df, amplitude=0.02):
    picos = []
    for date, row in df.iterrows():
        pico_alta = row['Forecast'] * (1 + amplitude)
        pico_baixa = row['Forecast'] * (1 - amplitude)
        picos.append({
            'data': date.strftime('%d/%m/%Y'),
            'pico_alta': pico_alta,
            'pico_baixa': pico_baixa
        })
    return picos

# Endpoint para retornar os picos de alta e baixa
@app.route('/pico-alta-baixa', methods=['GET'])
def obter_picos():
    last_data = X_scaled[-look_back:]
    n_future = 7  # Número de dias futuros
    future_predictions_scaled = forecast_future(model, last_data, look_back, n_future)
    future_predictions = scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1))

    # Criar DataFrame com as previsões futuras
    future_dates = [data['date'].max() + timedelta(days=i+1) for i in range(n_future)]
    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Forecast'])

    # Calcular os picos de alta e baixa
    picos = calcular_picos(future_df)

    # Retornar como JSON
    return jsonify(picos)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5050)
