#Stock-Price-Prediction-System

This repository implements a univariate and/or multivariate time series forecasting pipeline for stock price prediction using recurrent neural network architectures. The primary objective is to estimate future close prices from historical financial data with minimal prediction lag, supporting use cases such as algorithmic trading, trend analysis, and volatility forecasting.

Data Pipeline:

* Data source: Historical equity time series data retrieved via public APIs (e.g., Yahoo Finance, Alpha Vantage) or CSV-formatted datasets
* Features include:

  * OHLCV data (Open, High, Low, Close, Volume)
  * Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
* Missing value treatment: Time-based forward fill (pad) or linear interpolation
* Normalization: MinMax scaling to \[0, 1] range per feature dimension for stability in RNN training
* Dataset splitting: Chronologically partitioned into train (70%), validation (15%), and test (15%) sets without leakage

Sequence Modeling:

* Sliding window segmentation:

  * Look-back window size: w (e.g., 60 time steps)
  * Forecast horizon: h = 1 (next-day prediction) or configurable
  * Feature tensor shape: (batch\_size, w, n\_features)
* Optional multivariate encoding for multi-feature correlation capture

Model Architectures:

* LSTM-based regression model:

  * LSTM(units=64, return\_sequences=True)
  * Dropout(p=0.2)
  * LSTM(units=64, return\_sequences=False)
  * Dense(32) → ReLU
  * Dense(1) → Linear (for scalar regression)
* Alternative models optionally supported:

  * GRU
  * Bidirectional LSTM
  * Transformer Encoder (for non-recurrent modeling)
  * CNN1D for local temporal pattern extraction
* Loss function: Mean Squared Error (MSE)
* Optimizer: Adam (learning rate = 1e-4)

Training and Evaluation:

* Epochs: 50–200 with early stopping on validation loss
* Batch size: 32
* Callbacks:

  * ModelCheckpoint (monitor: val\_loss, save\_best\_only=True)
  * ReduceLROnPlateau (patience=5, factor=0.5)
* Evaluation metrics:

  * RMSE = sqrt(mean((y\_pred - y\_true)²))
  * MAE
  * R² score for goodness of fit
* Visualization:

  * Overlayed true vs predicted prices
  * Residual plots
  * Prediction confidence bands (via Monte Carlo Dropout or bootstrapping)

Deployment:

* Trained model serialized in HDF5 format (`model.h5`)
* Scaler object persisted via `joblib.dump()`
* Prediction pipeline supports real-time inference from streaming API data
* Flask/FastAPI REST API endpoint optionally implemented for remote model serving

Dependencies:

* Python 3.8+
* TensorFlow 2.x / Keras
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* yfinance (or equivalent) for data ingestion

Use Cases:

* Predictive analytics for portfolio managers
* Quantitative research and backtesting
* Input layer for reinforcement learning-based trading agents
