import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Função para criar janelas de dados
def criar_janelas(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Construir o modelo LSTM
def construir_modelo_lstm(look_back, input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, input_shape)))
    model.add(LSTM(units=50))
    model.add(Dense(1))  # Previsão de um valor contínuo
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Treinar o modelo LSTM
def treinar_modelo(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history
