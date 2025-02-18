import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def process_txt_file(file):
    # Handle Streamlit UploadedFile object or regular file object
    if isinstance(file, bytes):
        content = file.decode('utf-8')
    elif hasattr(file, 'getvalue'):  # Streamlit UploadedFile
        content = file.getvalue().decode('utf-8')
    else:
        content = file.read().decode('utf-8')

    lines = content.splitlines()
    data = [line.strip().split()[:24] for line in lines if line.strip() and not line.startswith('#')]
    
    columns = [
        "Routine Code", "Timestamp", "Routine Count", "Repetition Count", "Duration", "Integration Time [ms]",
        "Number of Cycles", "Saturation Index", "Filterwheel 1", "Filterwheel 2", "Zenith Angle [deg]", "Zenith Mode",
        "Azimuth Angle [deg]", "Azimuth Mode", "Processing Index", "Target Distance [m]",
        "Electronics Temp [°C]", "Control Temp [°C]", "Aux Temp [°C]", "Head Sensor Temp [°C]",
        "Head Sensor Humidity [%]", "Head Sensor Pressure [hPa]", "Scale Factor", "Uncertainty Indicator"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'].str.replace("T", " ").str.replace("Z", ""), 
        errors='coerce'
    )
    df_numeric = df.drop(columns=["Routine Code", "Timestamp"], errors='ignore').apply(pd.to_numeric, errors='coerce')
    
    return df, df_numeric



def load_and_preprocess_data(file):
    df, df_n = process_txt_file(file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    df_numeric = df.select_dtypes(include=[np.number])

    df_numeric.fillna(df_numeric.median(), inplace=True)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns, index=df.index)

    df_scaled["Timestamp"] = df["Timestamp"]

    return df, df_scaled


def train_anomaly_model(df_scaled):
    input_dim = df_scaled.shape[1] - 1
    encoding_dim = input_dim // 2
    
    autoencoder = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(encoding_dim, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid")
    ])
    
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(df_scaled.drop(columns=["Timestamp"], errors='ignore'),
                     df_scaled.drop(columns=["Timestamp"], errors='ignore'),
                     epochs=50, batch_size=64, validation_split=0.1, verbose=1)
    
    reconstructions = autoencoder.predict(df_scaled.drop(columns=["Timestamp"], errors='ignore'))
    reconstruction_errors = np.mean(np.abs(df_scaled.drop(columns=["Timestamp"], errors='ignore') - reconstructions), axis=1)
    
    threshold = np.percentile(reconstruction_errors, 99.9)
    df_scaled["Anomaly"] = (reconstruction_errors > threshold).astype(int)
    return df_scaled

def plot_data(df, df_scaled):
    columns_to_plot = [col for col in df.columns if col not in ["Timestamp", "Processed File"]]
    for column in columns_to_plot:
        fig = px.scatter(df, x="Timestamp", y=column, color=df_scaled["Anomaly"].map({0:'green', 1:'red'}),
                         color_discrete_map={"green": "green", "red": "red"},
                         title=f"{column} - Anomaly Detection")
        st.plotly_chart(fig)

def main():
    st.title("Deep Learning Anomaly Detection")
    uploaded_file = st.file_uploader("Upload L0 file", type=["txt"])
    
    if uploaded_file is not None:
        df, df_scaled = load_and_preprocess_data(uploaded_file)
        df_scaled = train_anomaly_model(df_scaled)
        plot_data(df, df_scaled)

if __name__ == "__main__":
    main()