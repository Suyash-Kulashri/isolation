import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import warnings
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import os
import boto3
import io
import botocore.exceptions

warnings.filterwarnings('ignore')

# AWS S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket-name")  # Load from environment variable, fallback to placeholder
S3_CSV_FILE = "newww.csv"
S3_PREDICTED_CSV_FILE = "new_predictions2.csv"
S3_MODEL_OUTPUT = "isolation_forest_model.pkl"
RANDOM_STATE = 42
CONTAMINATION = 0.1

# Initialize S3 client
try:
    s3_client = boto3.client('s3')
except botocore.exceptions.NoCredentialsError:
    st.error("AWS credentials not found. Please configure AWS credentials.")
    st.stop()

@st.cache_data
def load_and_preprocess_data(s3_bucket, s3_key):
    try:
        # Download CSV from S3
        obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"CSV file '{s3_key}' not found in S3 bucket '{s3_bucket}'.")
    except botocore.exceptions.ClientError as e:
        st.error(f"Error accessing S3: {e}")
        raise

    expected_columns = [
        'injection_time', 'system_name', 'column_serial_number', 'peak_width_5', 'retention_time', 
        'signal_to_noise_ratio', 'amount_percent', 'amount_value', 'area_percent', 'area_value', 
        'peak_width_50', 'resolution', 'analyte', 'method_set_name', 'project', 'sample_name', 
        'system_operator'
    ]
    available_columns = df.columns.tolist()
    missing_columns = [col for col in expected_columns if col not in available_columns]
    if missing_columns:
        st.warning(f"Missing columns: {missing_columns}. Using available columns: {available_columns}")

    if len(df.columns) != len(df.columns.unique()):
        st.warning("Duplicate column names detected in CSV. Removing duplicates...")
        df = df.loc[:, ~df.columns.duplicated()]

    if 'injection_time' not in df.columns:
        raise ValueError("Required column 'injection_time' is missing.")

    df['injection_time'] = pd.to_datetime(df['injection_time'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['column_serial_number', 'injection_time'])

    numeric_cols = [
        'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 'amount_value', 
        'area_percent', 'area_value', 'peak_width_50', 'resolution', 'peak_width_10'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df.groupby('column_serial_number')[col].transform(
                lambda x: x.fillna(x.median()) if x.median() == x.median() else x.fillna(0)
            )
            df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.99))

    df = df.sort_values(['column_serial_number', 'injection_time'])
    df['injection_count'] = df.groupby('column_serial_number').cumcount() + 1
    df['days_since_start'] = (df['injection_time'] - df.groupby('column_serial_number')['injection_time'].transform('min')).dt.days

    categorical_cols = [
        'system_name', 'column_serial_number', 'analyte', 'method_set_name', 'project', 
        'sample_name', 'system_operator'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_original'] = df[col]

    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders

def prepare_features(df, selected_param):
    feature_cols = [
        selected_param, 'injection_count', 'days_since_start',
        'resolution', 'retention_time', 'peak_width_5', 'peak_width_50',
        'system_name', 'analyte'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing feature columns: {missing_cols}")

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, feature_cols

def train_anomaly_model(X):
    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=100)
    model.fit(X)
    return model

def detect_anomalies(df, X, model, feature_cols, selected_param):
    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    df['anomaly_score'] = model.decision_function(X)
    
    Q1 = df[selected_param].quantile(0.25)
    Q3 = df[selected_param].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        st.warning(f"IQR for {selected_param} is 0. Using default thresholds.")
        upper_threshold = Q3 + 1.5
        lower_threshold = Q1 - 1.5
    else:
        upper_threshold = Q3 + 1.5 * IQR
        lower_threshold = Q1 - 1.5 * IQR
    
    df['anomaly_feature'] = selected_param
    df['anomaly_deviation'] = df[selected_param].apply(
        lambda x: x - upper_threshold if x > upper_threshold else 
                  lower_threshold - x if x < lower_threshold else 0
    )
    
    return df, {'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 'upper_threshold': upper_threshold, 'lower_threshold': lower_threshold}

def load_predicted_anomalies(s3_bucket, s3_key, selected_param, iqr_stats):
    try:
        # Download predicted CSV from S3
        obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except s3_client.exceptions.NoSuchKey:
        st.error(f"Predicted anomalies CSV file '{s3_key}' not found in S3 bucket '{s3_bucket}'.")
        return pd.DataFrame()
    except botocore.exceptions.ClientError as e:
        st.error(f"Error accessing S3: {e}")
        return pd.DataFrame()

    if 'predicted_date' not in df.columns:
        st.error("Required column 'predicted_date' missing in predicted anomalies CSV.")
        return pd.DataFrame()

    if f'predicted_{selected_param}' not in df.columns:
        st.error(f"Required column 'predicted_{selected_param}' not found in predicted data. Cannot proceed with anomaly detection.")
        return pd.DataFrame()

    df['predicted_date'] = pd.to_datetime(df['predicted_date'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['predicted_date'])

    required_cols = [
        f'predicted_{selected_param}', 'anomaly_flag', 'anomaly_cause', 
        'replacement_alert', 'column_serial_number', 'injection_count'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns in predicted anomalies CSV: {missing_cols}")

    if f'predicted_{selected_param}' in df.columns:
        upper_threshold = iqr_stats['upper_threshold']
        lower_threshold = iqr_stats['lower_threshold']
        df['anomaly_deviation'] = df[f'predicted_{selected_param}'].apply(
            lambda x: x - upper_threshold if x > upper_threshold else 
                      lower_threshold - x if x < lower_threshold else 0
        )
    else:
        df['anomaly_deviation'] = 0

    df['anomaly_feature'] = selected_param
    return df

def save_to_s3(obj, s3_bucket, s3_key):
    try:
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        s3_client.upload_fileobj(buffer, s3_bucket, s3_key)
    except botocore.exceptions.ClientError as e:
        st.error(f"Failed to upload model to S3: {e}")
        raise

def wrap_text(text, width=30):
    return '<br>'.join(textwrap.wrap(text, width=width))

def main():
    st.set_page_config(layout="wide")
    st.title("Chromatography Anomaly Detection Dashboard")

    try:
        df, label_encoders = load_and_preprocess_data(S3_BUCKET, S3_CSV_FILE)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.sidebar.header("Apply Filters")
    min_date = df['injection_time'].min().date() if 'injection_time' in df.columns else datetime(2024, 10, 5).date()
    max_date = df['injection_time'].max().date() if 'injection_time' in df.columns else datetime(2024, 10, 18).date()
    start_date = st.sidebar.date_input("Training Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("Training End Date", max_date, min_value=min_date, max_value=max_date)
    system_name = st.sidebar.selectbox("System Name", ["All"] + sorted(df['system_name_original'].unique().tolist()))
    method_set = st.sidebar.selectbox("Method Set", ["All"] + sorted(df['method_set_name_original'].unique().tolist()))
    selected_column = st.sidebar.multiselect("Column Serial Number", ["All"] + sorted(df['column_serial_number_original'].unique().tolist()), default=["All"])
    show_predicted_table = st.sidebar.checkbox("Show Predicted Data Table", value=False)
    show_deviation_graph = st.sidebar.checkbox("Show Deviation Graph", value=False)
    color_by = st.sidebar.selectbox("Color By", [
        'system_name', 'analyte', 'method_set_name', 'project', 'sample_name', 
        'system_operator'
    ])
    deviation_time_range = st.sidebar.selectbox("Deviation Time Range", ["One Day", "One Week", "One Month"], index=2)

    available_params = [
        col for col in [
            'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 
            'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 
            'peak_width_10'
        ] if col in df.columns
    ]
    selected_param = st.sidebar.selectbox("Performance Metric", available_params, index=0)

    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)
    prediction_start = end_date + timedelta(days=1)

    filtered_data = df[(df['injection_time'] >= start_date) & (df['injection_time'] <= end_date)]
    if system_name != "All":
        filtered_data = filtered_data[filtered_data['system_name_original'] == system_name]
    if method_set != "All":
        filtered_data = filtered_data[filtered_data['method_set_name_original'] == method_set]
    if "All" not in selected_column and selected_column:
        filtered_data = filtered_data[filtered_data['column_serial_number_original'].isin(selected_column)]

    if filtered_data.empty:
        st.error("No data found for the selected date range. Please adjust the Training Start Date and Training End Date to include historical data.")
        return

    filtered_data = filtered_data.loc[:, ~filtered_data.columns.duplicated()]

    try:
        X, scaler, feature_cols = prepare_features(filtered_data, selected_param)
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return

    iso_model = train_anomaly_model(X)
    filtered_data, iqr_stats = detect_anomalies(filtered_data, X, iso_model, feature_cols, selected_param)
    save_to_s3({'model': iso_model, 'scaler': scaler, 'label_encoders': label_encoders}, S3_BUCKET, S3_MODEL_OUTPUT)

    historical_values = filtered_data[selected_param].values
    anomaly_threshold = np.mean(historical_values) + 3 * np.std(historical_values)

    future_anomalies = load_predicted_anomalies(S3_BUCKET, S3_PREDICTED_CSV_FILE, selected_param, iqr_stats)

    fig = go.Figure()

    color_col = color_by + '_original' if f'{color_by}_original' in filtered_data.columns else color_by
    unique_values = filtered_data[color_col].astype(str).unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}

    for val in unique_values:
        subset = filtered_data[filtered_data[color_col] == val]
        fig.add_trace(go.Scatter(
            x=subset['injection_time'],
            y=subset[selected_param],
            mode='markers',
            name=f'{color_by}: {val}',
            marker=dict(color=color_map[val]),
            showlegend=False,
            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{selected_param}</b>: %{{y:.3f}}<br><b>{color_by}</b>: %{{customdata}}<extra></extra>',
            customdata=subset[color_col]
        ))

    historical_anomalies = filtered_data[filtered_data['anomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=historical_anomalies['injection_time'],
        y=historical_anomalies[selected_param],
        mode='markers',
        name='Historical Anomalies',
        marker=dict(color='orange', symbol='x'),
        text=historical_anomalies.apply(
            lambda row: (
                f"Date: {row['injection_time']}<br>"
                f"Parameter: {row[selected_param]:.3f}<br>"
                f"Anomaly: Yes<br>"
                f"{color_by}: {str(row[color_col]) if pd.notnull(row[color_col]) else 'Unknown'}<br>"
                f"Feature: {str(row['anomaly_feature']) if pd.notnull(row['anomaly_feature']) else selected_param}<br>"
                f"Deviation: {f'{float(row['anomaly_deviation']):.3f}' if pd.notnull(row['anomaly_deviation']) else '0.000'}<br>"
                f"Severity: {'Severe' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < -0.05 else 'Moderate' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < 0 else 'Normal'}"
            ), axis=1
        ),
        hovertemplate='%{text}<extra></extra>',
        hoverlabel=dict(bgcolor='white', font_size=12, align='left')
    ))

    if not future_anomalies.empty:
        pred_color_col = color_by + '_original' if f'{color_by}_original' in future_anomalies.columns else color_by
        unique_pred_values = future_anomalies[pred_color_col].unique() if pred_color_col in future_anomalies.columns else []
        for val in unique_pred_values:
            subset = future_anomalies[future_anomalies[pred_color_col] == val]
            fig.add_trace(go.Scatter(
                x=subset['predicted_date'],
                y=subset[f'predicted_{selected_param}'],
                mode='markers',
                name=f'Predicted {color_by}: {val}',
                visible='legendonly',
                marker=dict(color=color_map[val] if val in color_map else '#888888'),
                showlegend=False,
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{selected_param}</b>: %{{y:.3f}}<br><b>{color_by}</b>: %{{customdata}}<extra></extra>',
                customdata=subset[pred_color_col]
            ))

        predicted_anomalies = future_anomalies[future_anomalies['anomaly_flag']]
        fig.add_trace(go.Scatter(
            x=predicted_anomalies['predicted_date'],
            y=predicted_anomalies[f'predicted_{selected_param}'],
            mode='markers',
            name='Predicted Anomalies',
            marker=dict(color='red', symbol='x'),
            hovertemplate=(
                f'<b>Date</b>: %{{x}}<br>'
                f'<b>{selected_param}</b>: %{{y:.3f}}<br>'
                f'<b>Anomaly</b>: Yes<br>'
                f'<b>{color_by}</b>: %{{customdata[0]}}<br>'
                f'<b>Feature</b>: %{{customdata[1]}}<br>'
                f'<b>Deviation</b>: %{{customdata[2]:.3f}}<br>'
                f'<b>Cause</b>: %{{customdata[3]}}<br>'
                f'<b>Severity</b>: %{{customdata[4]}}<extra></extra>'
            ),
            customdata=predicted_anomalies[[pred_color_col, 'anomaly_feature', 'anomaly_deviation', 'anomaly_cause', 'anomaly_score']].apply(
                lambda row: [
                    row[pred_color_col] if pred_color_col in predicted_anomalies.columns else row['column_serial_number'],
                    row['anomaly_feature'],
                    row['anomaly_deviation'] if pd.notnull(row['anomaly_deviation']) else 0,
                    wrap_text(row['anomaly_cause']) if pd.notnull(row['anomaly_cause']) else 'Unknown',
                    '███' if row['anomaly_score'] < -0.05 else '█' if row['anomaly_score'] < 0 else ''
                ], axis=1
            ) if all(col in predicted_anomalies.columns for col in [pred_color_col, 'anomaly_feature', 'anomaly_deviation', 'anomaly_cause', 'anomaly_score']) else predicted_anomalies.apply(
                lambda row: [
                    row[pred_color_col] if pred_color_col in predicted_anomalies.columns else row['column_serial_number'],
                    selected_param,
                    0,
                    'Unknown',
                    ''
                ], axis=1
            ),
            hoverlabel=dict(namelength=-1, font_size=12, align='left')
        ))

    fig.update_layout(
        title=f'Predicted {selected_param} with Anomalies',
        xaxis_title='Injection Time',
        yaxis_title=selected_param,
        hovermode='closest',
        showlegend=True,
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0, orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display unique values with color dots
    unique_items = sorted(filtered_data[color_col].astype(str).unique().tolist())
    st.write(f"**{color_by} Values:**")
    html_items = ""
    for item in unique_items:
        color = color_map[item]
        html_items += f'<span style="color:{color};font-size:16px;margin-right:5px;">●</span><span style="margin-right:15px;">{item}</span>'

    st.markdown(f'<div>{html_items}</div>', unsafe_allow_html=True)

    st.subheader("Historical Data Table")
    columns_to_display = list(dict.fromkeys([
        'injection_time', selected_param, 'resolution', 'retention_time', 
        'anomaly_feature', 'anomaly_deviation'
    ]))
    st.dataframe(filtered_data[columns_to_display])

    if show_predicted_table and not future_anomalies.empty:
        st.subheader("Predicted Data Table")
        columns_to_display_pred = list(dict.fromkeys([
            'predicted_date', f'predicted_{selected_param}', 'anomaly_flag', 
            'anomaly_cause', 'replacement_alert', 'anomaly_deviation'
        ]))
        columns_to_display_pred = [col for col in columns_to_display_pred if col in future_anomalies.columns]
        st.dataframe(future_anomalies[columns_to_display_pred])

    if show_deviation_graph:
        time_range_label = deviation_time_range
        if deviation_time_range == "One Day":
            time_delta = timedelta(days=1)
        elif deviation_time_range == "One Week":
            time_delta = timedelta(days=7)
        else:
            time_delta = timedelta(days=30)

        latest_historical = filtered_data['injection_time'].max() if not filtered_data.empty else pd.Timestamp.now()
        latest_predicted = future_anomalies['predicted_date'].max() if not future_anomalies.empty else latest_historical
        latest_date = max(latest_historical, latest_predicted)

        deviation_data = filtered_data[
            (filtered_data['anomaly_deviation'] != 0) & 
            (filtered_data['injection_time'] >= latest_date - time_delta)
        ]
        pred_deviation_data = future_anomalies[
            (future_anomalies['anomaly_deviation'] != 0) & 
            (future_anomalies['predicted_date'] >= latest_date - time_delta)
        ] if not future_anomalies.empty else pd.DataFrame()

        st.subheader(f"Deviation of {selected_param} Over Time (Last {time_range_label}, Historical and Predicted)")
        
        fig_deviation = go.Figure()

        if not deviation_data.empty:
            fig_deviation.add_trace(go.Scatter(
                x=deviation_data['injection_time'],
                y=deviation_data['anomaly_deviation'],
                mode='markers',
                name='Historical Deviations',
                marker=dict(color='blue', size=8),
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Deviation</b>: %{{y:.3f}}<extra></extra>'
            ))

        if not pred_deviation_data.empty:
            fig_deviation.add_trace(go.Scatter(
                x=pred_deviation_data['predicted_date'],
                y=pred_deviation_data['anomaly_deviation'],
                mode='markers',
                name='Predicted Deviations',
                marker=dict(color='red', size=8),
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Deviation</b>: %{{y:.3f}}<extra></extra>'
            ))

        if not deviation_data.empty or not pred_deviation_data.empty:
            fig_deviation.update_layout(
                title=f'Deviation of {selected_param} Over Time (Last {time_range_label}, Historical and Predicted)',
                xaxis_title='Date',
                yaxis_title='Anomaly Deviation',
                hovermode='closest',
                showlegend=True
            )
            st.plotly_chart(fig_deviation, use_container_width=True)
        else:
            st.info(f"No deviations found for {selected_param} in the last {time_range_label}.")

    st.text(f"Showing historical data for {len(filtered_data)} injections")
    st.text(f"Showing predictions for {len(future_anomalies)} future dates starting from {prediction_start.strftime('%Y-%m-%d')}")

    if not future_anomalies.empty:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("OpenAI API key not found. Skipping anomaly summary generation.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                anomaly_summary = future_anomalies[future_anomalies['anomaly_flag']]
                if not anomaly_summary.empty:
                    summary_text = f"Summary of future anomalies for {selected_param} starting from {prediction_start.strftime('%Y-%m-%d')}:\n"
                    for index, row in anomaly_summary.iterrows():
                        summary_text += (
                            f"- Date: {row['predicted_date']}, "
                            f"Predicted Value: {row[f'predicted_{selected_param}']:.3f}, "
                            f"Cause: {row['anomaly_cause']}, "
                            f"Replacement Alert: {row['replacement_alert']}\n"
                        )
                    summary_text += "Please review these predictions and take action if necessary."

                    response = client.chat.completions.create(
                        model="gpt-4o",  # Use a valid model
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a chromatography expert summarizing anomaly predictions. "
                                    "Provide a concise summary in tabular format, including Date, Parameter, Cause, and Replacement Alert. "
                                    "Ensure causes are specific to chromatography (e.g., column clogging for high peak width, contamination for high retention time) "
                                    "and include actionable recommendations. "
                                    "If no data is available for a field, exclude it rather than showing NaN or empty values."
                                )
                            },
                            {"role": "user", "content": f"Summarize the following anomaly data: {summary_text}"}
                        ],
                        max_tokens=500
                    )

                    st.subheader("Future Anomaly Summary")
                    st.write(response.choices[0].message.content)
                else:
                    st.warning("No future anomalies detected. Check model sensitivity or data range.")
            except openai.error.AuthenticationError:
                st.warning("Invalid OpenAI API key. Skipping anomaly summary generation.")
            except openai.error.InvalidRequestError:
                st.warning("Invalid OpenAI model specified. Skipping anomaly summary generation.")
            except Exception as e:
                st.warning(f"Error generating anomaly summary: {e}")

if __name__ == "__main__":
    main()
