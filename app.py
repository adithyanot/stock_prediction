# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# --------------------------
# Helper Functions
# --------------------------
def SMA(series, window):
    return series.rolling(window).mean()

def EMA(series, window):
    return series.ewm(span=window, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, window=20, num_sd=2):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + (sd * num_sd)
    lower = ma - (sd * num_sd)
    return upper, lower

def prepare_features(df):
    df = df.copy()

    # --- Sanity check ---
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    # --- Technical indicators ---
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = RSI(df['Close'])
    df['EMA_20'] = EMA(df['Close'], 20)
    df['BB_Upper'], df['BB_Lower'] = bollinger_bands(df['Close'])

    macd_line, signal_line, hist = MACD(df['Close'])
    df['MACD_Line'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist

    # --- Target variable ---
    # Use single brackets so it's a Series, not a DataFrame
    df['Close_next'] = df['Close'].shift(-1)

    # Align to ensure proper comparison
    df['Close'], df['Close_next'] = df['Close'].align(df['Close_next'], axis=0)

    # Compute binary target safely
    df['Target'] = (df['Close_next'] > df['Close']).astype(int)

    # Drop rows with NaN values (mainly last one)
    df.dropna(inplace=True)

    return df


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Movement Prediction Dashboard")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker (e.g. RELIANCE.NS, TCS.NS)", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=730))
end_date = st.sidebar.date_input("End Date", datetime.now())
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

# --------------------------
# Data Loading
# --------------------------
st.subheader(f"📊 Historical Data for {ticker}")
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if data.empty:
    st.error("No data found for the given ticker.")
    st.stop()

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
data.reset_index(inplace=True)

# Price chart
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
fig_price.add_trace(go.Scatter(x=data['Date'], y=SMA(data['Close'], 20), mode='lines', name='SMA 20', line=dict(dash='dot')))
fig_price.update_layout(title="Stock Close Price with SMA 20", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_price, use_container_width=True)

# --------------------------
# Feature Engineering
# --------------------------
df_features = prepare_features(data)
drop_cols = ['Date', 'Close_next', 'Target']
X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns], errors='ignore')
y = df_features['Target']

# Time-based split
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Model Training
# --------------------------
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=200, random_state=42)
else:
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

with st.spinner("Training model..."):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

# --------------------------
# Evaluation
# --------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {acc:.4f} | **Precision:** {prec:.4f} | **Recall:** {rec:.4f} | **F1 Score:** {f1:.4f}")

cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
st.plotly_chart(cm_fig, use_container_width=True)

# --------------------------
# Prediction Visualization
# --------------------------
df_test = df_features.iloc[split:].copy()
df_test['Pred'] = y_pred
df_test['Pred_Direction'] = df_test['Pred'].map({1: "Up", 0: "Down"})

pred_fig = go.Figure()
pred_fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'], mode='lines', name='Close Price'))
pred_fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'], mode='markers',
                              marker=dict(color=df_test['Pred'], colorscale='RdBu', size=8),
                              name='Predicted Direction'))
pred_fig.update_layout(title="Actual Prices with Predicted Directions", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(pred_fig, use_container_width=True)

st.subheader("🔍 Recent Predictions")
st.dataframe(df_test[['Date', 'Close', 'Pred_Direction']].tail(10))
