# Time-Series-Forecasting-For-The-NIFTY50-Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("Nifty.csv")
scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])

look_back=10
def create_lagged_features(df, look_back):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[['Open', 'High', 'Low', 'Close']].iloc[i:i + look_back].values)
        y.append(df['Close'].iloc[i + look_back])
    return np.array(X), np.array(y)

X, y = create_lagged_features(df, look_back)

# Step 3: Split the data into training and testing sets (keeping the order)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Build the LSTM model
model = tf.keras.Sequential([
    LSTM(64, input_shape=(look_back, 4)),  # LSTM layer to handle 3D input
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Step 5: Train the model
model.fit(X_train, y_train, epochs=70, verbose=1)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean absolute Error: {mae:.4f}")

# Step 8: Visualize the results
plt.figure(figsize=(10, 4))
plt.plot(y_test, label='Actual Close Price')
plt.plot(y_pred, label='Predicted Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Days')
plt.ylabel('Normalized Close Price')
plt.legend()
plt.show()
