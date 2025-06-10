# vector.py
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def vectorize_and_store_data():
    # File paths for CSVs
    historical_file = "data_demo.csv"  # Path to historical data CSV
    predictions_file = "prediction_demo.csv"  # Path to predictions CSV
    
    # Load CSV files
    try:
        historical_df = pd.read_csv(historical_file)
        predictions_df = pd.read_csv(predictions_file)
    except FileNotFoundError as e:
        print(f"Error: CSV file not found - {e}")
        raise
    
    # Define columns for historical and predictions data
    historical_columns = [
        "injection_time", "injection_date", "project", "system_name", "sample_set_name",
        "method_set_name", "column_serial_number", "column_name", "analyte", "system_operator",
        "file_path", "created_at", "ids_file_id", "manual_integration", "channel_id",
        "channel_name", "peak_start", "peak_end", "amount_percent", "amount_value",
        "area_value", "area_percent", "resolution", "retention_time", "signal_to_noise_ratio",
        "symmetry_factor", "usp_tailing", "asym_at_10", "peak_width_5", "peak_width_10",
        "peak_width_50", "asymmetry_aia", "asymmetry_usp", "source_type", "sample_name",
        "sample_type", "injection_duration"
    ]
    predictions_columns = [
        "injection_time", "column_serial_number", "column_serial_number_original", "peak_width_5",
        "anomaly", "anomaly_score", "anomaly_feature", "anomaly_deviation", "anomaly_cause",
        "injection_count", "system_name", "system_name_original", "analyte", "analyte_original",
        "method_set_name", "method_set_name_original", "sample_name", "sample_name_original",
        "project", "project_original", "system_operator", "system_operator_original",
        "predicted_peak_width_5", "retention_time", "predicted_retention_time",
        "signal_to_noise_ratio", "predicted_signal_to_noise_ratio", "amount_percent",
        "predicted_amount_percent", "amount_value", "predicted_amount_value", "area_percent",
        "predicted_area_percent", "area_value", "predicted_area_value", "peak_width_50",
        "predicted_peak_width_50", "resolution", "predicted_resolution", "peak_width_10",
        "predicted_peak_width_10", "parameter", "predicted_date", "anomaly_flag", "replacement_alert"
    ]
    
    # Check for missing columns
    missing_hist = [col for col in historical_columns if col not in historical_df.columns]
    missing_pred = [col for col in predictions_columns if col not in predictions_df.columns]
    if missing_hist or missing_pred:
        print(f"Missing columns in historical data: {missing_hist}")
        print(f"Missing columns in predictions data: {missing_pred}")
        raise ValueError("Missing columns in CSV files")
    
    # Combine columns into a single string per row for vectorization
    historical_texts = historical_df[historical_columns].astype(str).agg(' '.join, axis=1).tolist()
    predictions_texts = predictions_df[predictions_columns].astype(str).agg(' '.join, axis=1).tolist()
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Vectorize the data
    historical_embeddings = model.encode(historical_texts, show_progress_bar=True)
    predictions_embeddings = model.encode(predictions_texts, show_progress_bar=True)
    
    # Create FAISS indexes
    dimension = historical_embeddings.shape[1]  # Embedding dimension
    historical_index = faiss.IndexFlatL2(dimension)
    predictions_index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to FAISS indexes
    historical_index.add(np.array(historical_embeddings, dtype='float32'))
    predictions_index.add(np.array(predictions_embeddings, dtype='float32'))
    
    # Save indexes and DataFrames to local disk
    faiss.write_index(historical_index, "historical_index.faiss")
    faiss.write_index(predictions_index, "predictions_index.faiss")
    historical_df.to_pickle("historical_df.pkl")
    predictions_df.to_pickle("predictions_df.pkl")
    
    print("Vectorization complete. FAISS indexes and DataFrames saved locally.")
    return model

if __name__ == "__main__":
    vectorize_and_store_data()