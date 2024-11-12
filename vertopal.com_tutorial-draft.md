---
jupyter:
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
## 1. Introduction {#1-introduction}

As part of the SEDIMARK toolbox that Users will use for configuring AI
and Data Processing pipelines for their use case, AI tasks such as
forecasting are made readily available for inferencing on Data Assets.
Time series forecasting has a wide range of applications across various
fields, including financial market prediction, weather forecasting, and
traffic flow prediction.

In this tutorial, we will use Python to demonstrate the basic AI
workflow for time series forecasting, specifically focusing on
temperature forecasting for agriculture use cases. Accurate temperature
forecasting is crucial for agriculture as it helps farmers plan their
activities, manage crops, and optimize yields.
:::

::: {.cell .markdown}
## 2. Environment Setup {#2-environment-setup}

We need to install some toolboxes and libraries for this experiment.
Therefore, please copy and use the below command in your python
terminal:

    pip install numpy pandas matplotlib scikit-learn torch
:::

::: {.cell .markdown}
## 3. Data Preprocessing {#3-data-preprocessing}

In this section, we generate the simulation data and apply the
preprocessing.
:::

::: {.cell .code}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generate sample data
date_rng = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df['temperature'] = np.random.randint(20, 35, size=(len(date_rng)))

# Set date as index
df.set_index('date', inplace=True)

# Visualize data
df['temperature'].plot(figsize=(12, 6), title='Temperature Time Series')
plt.show()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df['temperature_scaled'] = scaler.fit_transform(df['temperature'].values.reshape(-1, 1))

# Split into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Create dataset for Transformer
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_train, y_train = create_dataset(train['temperature_scaled'].values, time_step)
X_test, y_test = create_dataset(test['temperature_scaled'].values, time_step)

# Convert to PyTorch tensors
import torch
X_train = torch.tensor(X_train.reshape(X_train.shape[0], time_step, 1), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test.reshape(X_test.shape[0], time_step, 1), dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
```
:::

::: {.cell .markdown}
## 4. Build the Simple Transformer Model {#4-build-the-simple-transformer-model}
:::

We use the toolbox and librires support provided by Pytorch to create a simple and basic Transformer model (Encoder-Decoder).

::: {.cell .code}
``` python
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, num_heads, d_model, num_encoder_layers, num_decoder_layers, dff):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dff)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dff)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(d_model * time_step, dff)
        self.dense2 = nn.Linear(dff, 1)

    def forward(self, src):
        encoder_output = self.transformer_encoder(src)
        decoder_output = self.transformer_decoder(encoder_output, encoder_output)
        flatten_output = self.flatten(decoder_output)
        dense_output = self.dense1(flatten_output)
        output = self.dense2(dense_output)
        return output

# Hyperparameters
num_heads = 2
d_model = 64
num_encoder_layers = 2
num_decoder_layers = 2
dff = 128

# Create model
model = TransformerModel(num_heads, d_model, num_encoder_layers, num_decoder_layers, dff)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 50
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```
:::

::: {.cell .markdown}
## 5. Model Evaluation {#5-model-evaluation}
:::

We evaluate our trained model on the created data.

::: {.cell .code}
``` python
import math
from sklearn.metrics import mean_squared_error

model.eval()
with torch.no_grad():
    train_predict = model(X_train).squeeze().numpy()
    test_predict = model(X_test).squeeze().numpy()

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_score = math.sqrt(mean_squared_error(y_train, train_predict))
test_score = math.sqrt(mean_squared_error(y_test, test_predict))
print(f'Train Score: {train_score} RMSE')
print(f'Test Score: {test_score} RMSE')

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(df['temperature'], label='Actual Data')
plt.plot(df.index[time_step:train_size], train_predict, label='Train Predict')
plt.plot(df.index[train_size+time_step+1:], test_predict, label='Test Predict')
plt.legend()
plt.show()
```
:::

::: {.cell .markdown}
## 6. Conclusion {#6-conclusion}

This tutorial demonstrates how to use a basic Transformer model for time
series forecasting, specifically for temperature prediction in
agriculture. Accurate temperature forecasting is essential for
agricultural planning and decision-making, helping farmers optimize crop
management and improve yields. Through this example, readers can gain a
fundamental understanding of applying Transformers to time series
forecasting and further research and optimize the model for better
prediction performance.
:::
