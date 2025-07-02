import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from tensorflow.keras.models import load_model
import joblib
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
import matplotlib.pyplot as plt
import os

# ---------------------- Load model and scaler ----------------------
model = load_model('model1/best_model.h5', compile=False)
scaler = joblib.load('model1/btc_scaler.pkl')

# ---------------------- Fetch BTC data ----------------------
def fetch_btc_data():
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {'fsym': 'BTC', 'tsym': 'USD', 'limit': 400}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return pd.DataFrame()
    data = response.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['Open time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Karachi')
    df.rename(columns={'volumefrom': 'Volume'}, inplace=True)
    df = df[['Open time', 'open', 'high', 'low', 'close', 'Volume']]
    df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df

# ---------------------- Add indicators ----------------------
def add_indicators(df):
    df['rsi'] = RSIIndicator(close=df['Close']).rsi()
    df['macd'] = MACD(close=df['Close']).macd()
    df['ema'] = EMAIndicator(close=df['Close'], window=14).ema_indicator()
    df['stoch_k'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    return df.dropna()

# ---------------------- Make prediction ----------------------
def predict_next_close(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'ema', 'stoch_k']
    last_100 = df[features].values[-100:]
    scaled = scaler.transform(last_100)
    X = np.expand_dims(scaled, axis=0)
    pred_scaled = model.predict(X)
    
    close_idx = features.index('Close')
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[:, close_idx] = pred_scaled
    predicted_close = scaler.inverse_transform(dummy)[:, close_idx][0]

    return predicted_close

# ---------------------- Signal generation ----------------------
def generate_signal(predicted, actual):
    if predicted > actual * 1.005:
        return "🟢 BUY", "green"
    elif predicted < actual * 0.995:
        return "🔴 SELL", "red"
    else:
        return "🟡 HOLD", "orange"

# ---------------------- Logging ----------------------
def log_prediction(pred_time, actual, predicted, change_pct, signal):
    log_file = "btc_predictions_log.csv"
    log_data = {
        "Predicted Time": pred_time.strftime('%Y-%m-%d %H:%M %Z'),
        "Actual Close (t)": actual,
        "Predicted Close (t+1)": predicted,
        "Change (%)": change_pct,
        "Signal": signal
    }
    df_log = pd.DataFrame([log_data])
    if not os.path.exists(log_file):
        df_log.to_csv(log_file, index=False)
    else:
        df_log.to_csv(log_file, mode='a', header=False, index=False)

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="BTC Next-Hour Predictor", layout="wide")
st.title("📈 Bitcoin (BTC) Next-Hour Price Prediction")
st.markdown("Predicting the next hour's BTC close price using LSTM + technical indicators.")

placeholder = st.empty()

df = fetch_btc_data()
if df.empty:
    st.error("⚠️ Failed to fetch BTC data from API.")
else:
    df = add_indicators(df)
    if len(df) < 100:
        st.warning("⚠️ Not enough data to make prediction.")
    else:
        actual_close_t = df['Close'].iloc[-1]
        actual_time_t = df['Open time'].iloc[-1]
        predicted_close_t1 = predict_next_close(df)
        predicted_time_t1 = actual_time_t + timedelta(hours=1)

        change_pct = ((predicted_close_t1 - actual_close_t) / actual_close_t) * 100
        signal, color = generate_signal(predicted_close_t1, actual_close_t)

        # Log this prediction
        log_prediction(predicted_time_t1, actual_close_t, predicted_close_t1, change_pct, signal)

        # Display metrics
        with placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🕒 Predicted Time (Asia/Karachi)", predicted_time_t1.strftime('%Y-%m-%d %H:%M %Z'))
            col2.metric("📉 Actual Close (t)", f"${actual_close_t:,.2f}")
            col3.metric("🔮 Predicted Close (t+1)", f"${predicted_close_t1:,.2f}")
            col4.metric("📈 Change (%)", f"{change_pct:+.2f}%")

            st.markdown(f"### Signal: <span style='color:{color}; font-size:24px'>{signal}</span>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df['Open time'], df['Close'], label="Actual Close", color='blue')
            ax.scatter([predicted_time_t1], [predicted_close_t1], color=color, s=100, label="Predicted Close (t+1)")
            ax.set_title("BTC Actual Close + Predicted Next-Hour Close")
            ax.set_xlabel("Time (Asia/Karachi)")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
