import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

# Função para previsão futura
def forecast_future(model, last_data, look_back, n_future, input_shape):
    future_predictions = []
    current_step = last_data.reshape(1, look_back, input_shape)

    for _ in range(n_future):
        pred = model.predict(current_step)
        future_predictions.append(pred[0, 0])
        current_step = np.roll(current_step, -1, axis=1)
        current_step[0, -1, 0] = pred[0, 0]

    return np.array(future_predictions)

# Função para simular picos de alta e baixa
def calcular_picos(df, amplitude=0.02):
    picos = []
    for date, row in df.iterrows():
        pico_alta = row['Forecast'] * (1 + amplitude)
        pico_baixa = row['Forecast'] * (1 - amplitude)
        picos.append(f"Pico de alta para o dia {date.strftime('%d/%m/%Y')}: {pico_alta:.2f}")
        picos.append(f"Pico de baixa para o dia {date.strftime('%d/%m/%Y')}: {pico_baixa:.2f}")
    return picos

# Função para plotar os resultados
def plot_results(data, future_df):
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['close'], label='Histórico', color='blue')
    plt.plot(future_df.index, future_df['Forecast'], label='Previsão Futuras', color='red')
    plt.xlabel('Data')
    plt.ylabel('Preço do Bitcoin')
    plt.title('Previsão de Preço do Bitcoin com LSTM')
    plt.legend()
    plt.show()
