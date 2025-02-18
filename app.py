# Streamlit Anomaly Detection App with Dependency Fixes

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras import models
import subprocess
import sys

# Ensure pip and packages are up-to-date
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'pygments', 'markdown-it-py', 'rich'], check=True)
except Exception as e:
    st.error(f"Dependency installation error: {e}")

# Load trained model and scaler
try:
    loaded_autoencoder = models.load_model('autoencoder_model.h5', 
                                           custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

def process_txt_file(file):
    lines = file.readlines()
    data = [line.strip().split()[:24] for line in lines if line.strip() and not line.startswith('#')]
    columns = [
        "Routine Code", "Timestamp", "Routine Count", "Repetition Count", "Duration", "Integration Time [ms]",
        "Number of Cycles", "Saturation Index", "Filterwheel 1", "Filterwheel 2", "Zenith Angle [deg]", "Zenith Mode",
        "Azimuth Angle [deg]", "Azimuth Mode", "Processing Index", "Target Distance [m]",
        "Electronics Temp [°C]", "Control Temp [°C]", "Aux Temp [°C]", "Head Sensor Temp [°C]",
        "Head Sensor Humidity [%]", "Head Sensor Pressure [hPa]", "Scale Factor", "Uncertainty Indicator"
    ]
    df = pd.DataFrame(data, columns=columns)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.replace("T", " ").str.replace("Z", ""), errors='coerce')
    df_numeric = df.drop(columns=["Routine Code", "Timestamp"], errors='ignore').apply(pd.to_numeric, errors='coerce')
    return df, df_numeric

def detect_anomalies(df_numeric):
    df_scaled = scaler.transform(df_numeric)
    reconstructions = loaded_autoencoder.predict(df_scaled)
    errors = np.mean(np.abs(df_scaled - reconstructions), axis=1)
    threshold = np.percentile(errors, 99.9)
    return errors > threshold

def plot_results(df, anomalies):
    normal_data, anomalous_data = df[~anomalies], df[anomalies]
    st.subheader("Anomaly Detection Results")
    for column in df.columns.drop('Timestamp', errors='ignore'):
        plt.figure(figsize=(12, 6))
        plt.plot(normal_data['Timestamp'], normal_data[column], label='Normal', alpha=0.5)
        plt.scatter(anomalous_data['Timestamp'], anomalous_data[column], color='red', label='Anomaly', marker='x')
        plt.title(f'{column} - Anomaly Detection')
        plt.xlabel('Timestamp')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

st.title("📊 Streamlit Anomaly Detection App")
uploaded_file = st.file_uploader("Upload a .txt file", type=['txt'])
if uploaded_file:
    df, df_numeric = process_txt_file(uploaded_file)
    anomalies = detect_anomalies(df_numeric)
    st.write(f"### Total Anomalies Detected: {anomalies.sum()}")
    plot_results(df, anomalies)
