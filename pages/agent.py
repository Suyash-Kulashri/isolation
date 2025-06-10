import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
st.set_page_config(layout="wide")

# Set the current time
CURRENT_TIME = datetime(2025, 6, 6, 16, 58)  # 04:58 PM IST on June 06, 2025

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()
client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    st.error("Pinecone API key not found. Please set it in the .env file.")
    st.stop()
pc = Pinecone(api_key=pinecone_api_key)

# Connect to Pinecone indexes
INDEX_NAMES = [
    "processed-chromatography-data",
    "lstm-prediction",
    "chromatography-data",
    "anomaly-result"
]
try:
    indexes = {name: pc.Index(name) for name in INDEX_NAMES}
except Exception as e:
    st.error(f"Error connecting to Pinecone indexes: {e}")
    st.stop()

# Load DataFrames (assuming metadata is stored separately)
try:
    historical_df = pd.read_pickle("historical_df.pkl")
    predictions_df = pd.read_pickle("predictions_df.pkl")
except Exception as e:
    st.error(f"Error loading DataFrames: {e}")
    st.stop()

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "message": "Hello! I'm your AI Assistant. I can help you analyze anomalies, interpret data patterns, and provide insights about your system performance. How can I assist you today?", "timestamp": CURRENT_TIME}
    ]

# Define columns for metadata
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

# Function to calculate time difference in a human-readable format
def time_ago(start_time):
    delta = CURRENT_TIME - start_time
    if delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    minutes = (delta.seconds % 3600) // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    return "Just now"

# Function to let LLM decide which Pinecone indexes to search
def decide_indexes_to_search(query):
    prompt = f"""
    You are an AI assistant specializing in chromatography data analysis. The user has asked the following query: "{query}".
    There are four Pinecone indexes available:
    - processed-chromatography-data: Contains historical chromatography data.
    - lstm-prediction: Contains predicted chromatography data using LSTM models.
    - chromatography-data: Contains additional historical chromatography data.
    - anomaly-result: Contains anomaly detection results for chromatography data.
    
    Based on the query, determine which indexes to search. A query might require searching one or more indexes.
    Return a list of index names to search, for example: ["processed-chromatography-data", "chromatography-data"].
    If the query is unrelated to the indexes, return an empty list: [].
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant that determines relevant data sources based on user queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        selected_indexes = eval(response.choices[0].message.content.strip())  # Safely evaluate the list
        # Validate the selected indexes
        valid_indexes = [idx for idx in selected_indexes if idx in INDEX_NAMES]
        return valid_indexes
    except Exception as e:
        st.error(f"Error deciding indexes: {e}")
        return []

# Function to search selected Pinecone indexes
def search_pinecone(query, indexes_to_search):
    if not indexes_to_search:
        return []
    
    query_embedding = model.encode([query])[0].tolist()  # Convert to list for Pinecone
    results = []
    
    for index_name in indexes_to_search:
        index = indexes[index_name]
        try:
            response = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            for item in response['matches']:
                result = item['metadata']
                result['distance'] = float(item['score'])
                result['source_index'] = index_name
                results.append(result)
        except Exception as e:
            st.error(f"Error searching index {index_name}: {e}")
    
    return results

# Function to generate response using OpenAI
def generate_response(query, search_results):
    context = "Relevant data:\n"
    for result in search_results:
        context += f"From {result['source_index']}: {str(result)}\n"

    prompt = f"User query: {query}\n\n{context}\n\nProvide a detailed and insightful response based on the query and the provided data."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in anomaly analysis and system insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# Custom CSS for styling
st.markdown("""
    <style>
    .main-container {
        background-color: #1E2A44;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .chat-container {
        background-color: #2A3555;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .chat-message {
        background-color: #3B4771;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .quick-questions {
        background-color: #2A3555;
        padding: 15px;
        border-radius: 10px;
    }
    .ai-capabilities {
        background-color: #2A3555;
        padding: 15px;
        border-radius: 10px;
    }
    .job-title {
        font-size: 18px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .quick-question-button {
        background-color: transparent;
        color: #63B3ED;
        border: none;
        text-align: left;
        padding: 5px 0;
        cursor: pointer;
    }
    .send-button {
        background: linear-gradient(90deg, #FF6B6B, #FF00FF);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Main layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1>Chromatography AI Agent</h1>", unsafe_allow_html=True)
st.markdown("<p>Intelligent assistant for anomaly analysis and system insights</p>", unsafe_allow_html=True)

# Main layout with two columns
col1, col2 = st.columns([3, 1])

with col1:
    # Chat section
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="job-title"><span style="margin-right: 8px;">🤖</span>AI Assistant</div>', unsafe_allow_html=True)

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="chat-message">{chat["message"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 12px; color: #A0AEC0;">{time_ago(chat["timestamp"])}</p>', unsafe_allow_html=True)

    # Chat input
    chat_input = st.text_input("Ask me about anomalies, patterns, or system insights...", key="chat_input")
    if st.button("Send", key="send_button"):
        if chat_input:
            st.session_state.chat_history.append({
                "role": "user",
                "message": chat_input,
                "timestamp": CURRENT_TIME
            })
            # Let LLM decide which indexes to search
            indexes_to_search = decide_indexes_to_search(chat_input)
            # Search selected Pinecone indexes
            search_results = search_pinecone(chat_input, indexes_to_search)
            # Generate response using OpenAI
            response = generate_response(chat_input, search_results)
            st.session_state.chat_history.append({
                "role": "assistant",
                "message": response,
                "timestamp": CURRENT_TIME
            })
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Quick Questions section
    st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
    st.markdown("<h3>Quick Questions</h3>", unsafe_allow_html=True)
    if st.button("Analyze recent anomaly patterns", key="qq1", help="Ask about anomaly patterns"):
        st.session_state.chat_history.append({
            "role": "user",
            "message": "Analyze recent anomaly patterns",
            "timestamp": CURRENT_TIME
        })
        indexes_to_search = decide_indexes_to_search("Analyze recent anomaly patterns")
        search_results = search_pinecone("Analyze recent anomaly patterns", indexes_to_search)
        response = generate_response("Analyze recent anomaly patterns", search_results)
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": CURRENT_TIME
        })
        st.rerun()
    if st.button("What caused the performance spike at 8 AM?", key="qq2"):
        st.session_state.chat_history.append({
            "role": "user",
            "message": "What caused the performance spike at 8 AM?",
            "timestamp": CURRENT_TIME
        })
        indexes_to_search = decide_indexes_to_search("What caused the performance spike at 8 AM?")
        search_results = search_pinecone("What caused the performance spike at 8 AM?", indexes_to_search)
        response = generate_response("What caused the performance spike at 8 AM?", search_results)
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": CURRENT_TIME
        })
        st.rerun()
    if st.button("Generate a security report", key="qq3"):
        st.session_state.chat_history.append({
            "role": "user",
            "message": "Generate a security report",
            "timestamp": CURRENT_TIME
        })
        indexes_to_search = decide_indexes_to_search("Generate a security report")
        search_results = search_pinecone("Generate a security report", indexes_to_search)
        response = generate_response("Generate a security report", search_results)
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": CURRENT_TIME
        })
        st.rerun()
    if st.button("Recommend optimization strategies", key="qq4"):
        st.session_state.chat_history.append({
            "role": "user",
            "message": "Recommend optimization strategies",
            "timestamp": CURRENT_TIME
        })
        indexes_to_search = decide_indexes_to_search("Recommend optimization strategies")
        search_results = search_pinecone("Recommend optimization strategies", indexes_to_search)
        response = generate_response("Recommend optimization strategies", search_results)
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": CURRENT_TIME
        })
        st.rerun()
    if st.button("Show me the top 5 system risks", key="qq5"):
        st.session_state.chat_history.append({
            "role": "user",
            "message": "Show me the top 5 system risks",
            "timestamp": CURRENT_TIME
        })
        indexes_to_search = decide_indexes_to_search("Show me the top 5 system risks")
        search_results = search_pinecone("Show me the top 5 system risks", indexes_to_search)
        response = generate_response("Show me the top 5 system risks", search_results)
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": CURRENT_TIME
        })
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # AI Capabilities section
    st.markdown('<div class="ai-capabilities">', unsafe_allow_html=True)
    st.markdown("<h3>AI Capabilities</h3>", unsafe_allow_html=True)
    st.markdown('<div class="job-title"><span style="margin-right: 8px;">📊</span>Pattern Analysis</div>', unsafe_allow_html=True)
    st.markdown("<p>Detect anomaly patterns</p>", unsafe_allow_html=True)
    st.markdown('<div class="job-title"><span style="margin-right: 8px;">⚡</span>Real-time Insights</div>', unsafe_allow_html=True)
    st.markdown("<p>Live data interpretation</p>", unsafe_allow_html=True)
    st.markdown('<div class="job-title"><span style="margin-right: 8px;">💬</span>Natural Language</div>', unsafe_allow_html=True)
    st.markdown("<p>Conversational interface</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)