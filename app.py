import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def process_txt_file(file):
    if hasattr(file, 'getvalue'):  # Streamlit UploadedFile
        raw_bytes = file.getvalue()
    else:
        raw_bytes = file.read()

    # Attempt decoding with common encodings
    for encoding in ['utf-8', 'ISO-8859-1', 'utf-16', 'windows-1252']:
        try:
            content = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not decode file using common encodings.")

    # Split lines and find the start of data
    lines = content.splitlines()

    # Find the index of the second separator line
    data_start_index = None
    separator_count = 0

    for i, line in enumerate(lines):
        if line.startswith('---'):
            separator_count += 1
            if separator_count == 2:
                data_start_index = i + 1
                break

    if data_start_index is None:
        st.error("Data section not found in the uploaded file.")
        return pd.DataFrame(), pd.DataFrame()

    # Extract relevant data lines only
    data_lines = [
    line.strip().split()[:24] 
    for line in lines[data_start_index:] 
    if line.strip() and not line.startswith('#') and "# INFO" not in line
]
    
    columns = [
        "Routine Code", "Timestamp", "Routine Count", "Repetition Count", "Duration", "Integration Time [ms]",
        "Number of Cycles", "Saturation Index", "Filterwheel 1", "Filterwheel 2", "Zenith Angle [deg]", "Zenith Mode",
        "Azimuth Angle [deg]", "Azimuth Mode", "Processing Index", "Target Distance [m]",
        "Electronics Temp [°C]", "Control Temp [°C]", "Aux Temp [°C]", "Head Sensor Temp [°C]",
        "Head Sensor Humidity [%]", "Head Sensor Pressure [hPa]", "Scale Factor", "Uncertainty Indicator"
    ]
    
    df = pd.DataFrame(data_lines, columns=columns)
    
    # Parse the Timestamp column safely
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'].str.replace("T", " ").str.replace("Z", ""), 
        errors='coerce'
    )
    
    # Remove rows with invalid timestamps
    df = df[df['Timestamp'].notnull()].reset_index(drop=True)
    
    # Convert numeric columns properly
    numeric_columns = [col for col in df.columns if col not in ["Routine Code", "Timestamp"]]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df_numeric = df.drop(columns=["Routine Code", "Timestamp"], errors='ignore')

    return df, df_numeric



def load_and_preprocess_data(file):
    df, df_n = process_txt_file(file)

    # Convert Timestamp column to datetime and sort
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Select only numeric columns and remove any non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Drop columns that have all NaN values
    df_numeric = df_numeric.dropna(axis=1, how='all')

    # Handle missing values by filling them with the median
    if not df_numeric.empty:
        df_numeric = df_numeric.fillna(df_numeric.median(numeric_only=True))

    # Drop columns with only one unique value (no variance)
    df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]


    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # If numeric dataframe is empty after cleaning, raise an error
    if df_numeric.empty:
        st.error("No valid numeric columns found. Please check the uploaded file.")
        raise ValueError("No valid numeric columns found in the uploaded file.")

    # Apply MinMaxScaler only if df_numeric is not empty
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns,
        index=df.index
    )

    # Add Timestamp back to the scaled DataFrame
    df_scaled["Timestamp"] = df["Timestamp"]

    return df, df_scaled

def train_anomaly_model(df_scaled):
    input_dim = df_scaled.shape[1] - 1  # Exclude "Timestamp"
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

    # Generate reconstructions
    reconstructions = autoencoder.predict(df_scaled.drop(columns=["Timestamp"], errors='ignore'))

    # Compute per-column reconstruction errors
    numeric_columns = df_scaled.drop(columns=["Timestamp"], errors='ignore').columns
    reconstruction_errors_per_column = np.abs(df_scaled[numeric_columns] - reconstructions)

    # Compute total anomaly score per row
    reconstruction_errors = reconstruction_errors_per_column.mean(axis=1)

    # Define anomaly threshold
    threshold = np.percentile(reconstruction_errors, 99.9)
    df_scaled["Anomaly"] = (reconstruction_errors > threshold).astype(int)

    # Identify top anomaly-causing columns
    anomaly_columns = []
    
    for i in range(len(df_scaled)):
        if df_scaled.loc[i, "Anomaly"] == 1:
            # Get column names of the top 3 highest reconstruction errors
            sorted_errors = reconstruction_errors_per_column.iloc[i].nlargest(3)
            max_error_columns = sorted_errors.index.tolist()  # Convert to list of column names
            anomaly_columns.append(", ".join(max_error_columns))
        else:
            anomaly_columns.append("")

    # Assign the anomaly-causing columns to the new column
    df_scaled["Anomaly_Column"] = anomaly_columns

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
        
        
        # Display the anomalies
        st.subheader("List of Anomalies Detected")
        anomalies = df[df_scaled["Anomaly"] == 1]
        if not anomalies.empty:
            st.dataframe(anomalies)
        else:
            st.write("No anomalies detected.")
        plot_data(df, df_scaled)


if __name__ == "__main__":
    main()