import streamlit as st
import time
from datetime import datetime, timedelta

# Set the current time as per the provided date and time
CURRENT_TIME = datetime(2025, 6, 6, 12, 56)  # 12:56 PM IST on June 06, 2025

st.set_page_config(layout='wide')

# Initialize session state for job executions
if 'job_executions' not in st.session_state:
    st.session_state.job_executions = []

# Sample job data
jobs = [
    {"name": "Job_0", "description": "Extract, transform, and load data from multiple sources", "status": "Idle", "duration": "45 min"},
    {"name": "Job_1", "description": "Train anomaly detection models with latest data", "status": "Idle", "duration": "2h 15min"},
    {"name": "Job_2", "description": "Generate daily analytics and insights reports", "status": "Idle", "duration": "12 min"},
    {"name": "Job_3", "description": "Validate data quality and integrity checks", "status": "Idle", "duration": "8 min"},
    {"name": "Job_4", "description": "Backup critical data and configurations", "status": "Idle", "duration": "1h 30min"},
    {"name": "Job_5", "description": "Process and analyze large datasets", "status": "Idle", "duration": "50 min"},
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

# Function to handle job run
def run_job(job):
    job["status"] = "Running"
    execution = {
        "name": job["name"],
        "status": "Running",
        "start_time": CURRENT_TIME,
        "duration": job["duration"]
    }
    st.session_state.job_executions.append(execution)
    # Simulate job completion after a delay (for demo purposes)
    time.sleep(1)  # Replace with actual job execution logic
    job["status"] = "Completed"
    execution["status"] = "Completed"

# Custom CSS for styling
st.markdown("""
    <style>
    .main-container {
        background-color: #1E2A44;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .job-card {
        background-color: #2A3555;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .job-title {
        font-size: 18px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .status-idle { color: #A0AEC0; }
    .status-running { color: #63B3ED; }
    .status-completed { color: #68D391; }
    .status-failed { color: #F56565; }
    .run-button {
        background: linear-gradient(90deg, #4A90E2, #9B59B6);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stop-button {
        background-color: #E53E3E;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
    }
    .table-header {
        font-weight: bold;
        color: white;
    }
    .table-row {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Main layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1>Jobs Management</h1>", unsafe_allow_html=True)
st.markdown("<p>Control and monitor your Databricks jobs</p>", unsafe_allow_html=True)

# Refresh All button
st.markdown('<button class="run-button" style="float: right;">Refresh All</button>', unsafe_allow_html=True)

# Check if any job is running
any_job_running = any(job["status"] == "Running" for job in jobs)

# Job cards
cols = st.columns(3)
for i, job in enumerate(jobs):
    with cols[i % 3]:
        st.markdown(f'<div class="job-card">', unsafe_allow_html=True)
        status_class = f"status-{job['status'].lower()}"
        st.markdown(f'<div class="job-title"><span style="margin-right: 8px;">📋</span>{job["name"]}<span style="margin-left: 8px;" class="{status_class}">● {job["status"]}</span></div>', unsafe_allow_html=True)
        st.markdown(f"<p>{job['description']}</p>", unsafe_allow_html=True)

        # Find the most recent execution for this job to display Last Run
        last_execution = next((exec_ for exec_ in reversed(st.session_state.job_executions) if exec_["name"] == job["name"]), None)
        last_run = time_ago(last_execution["start_time"]) if last_execution else "Never"
        st.markdown(f"<p><b>Last Run:</b> {last_run}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Duration:</b> {job['duration']}</p>", unsafe_allow_html=True)

        # Action buttons
        if job["status"] == "Running":
            st.markdown('<button class="stop-button">Stop</button>', unsafe_allow_html=True)
        else:
            # Disable the button if any job is running and this job is not running
            disabled = any_job_running
            if st.button("Run Job", key=f"run_{job['name']}", help="Run this job", disabled=disabled):
                run_job(job)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Recent Job Executions
st.markdown("<h2>Recent Job Executions</h2>", unsafe_allow_html=True)
if st.session_state.job_executions:
    st.markdown('<div class="table-header">', unsafe_allow_html=True)
    cols = st.columns([2, 1, 2, 1, 1])
    cols[0].write("Job Name")
    cols[1].write("Status")
    cols[2].write("Started")
    cols[3].write("Duration")
    cols[4].write("Actions")
    st.markdown('</div>', unsafe_allow_html=True)

    for execution in reversed(st.session_state.job_executions[-3:]):  # Show last 3 executions
        st.markdown('<div class="table-row">', unsafe_allow_html=True)
        cols = st.columns([2, 1, 2, 1, 1])
        cols[0].write(execution["name"])
        status_class = f"status-{execution['status'].lower()}"
        cols[1].markdown(f'<span class="{status_class}">● {execution["status"]}</span>', unsafe_allow_html=True)
        cols[2].write(time_ago(execution["start_time"]))
        cols[3].write(execution["duration"])
        cols[4].write("View Logs")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("<p>No recent executions.</p>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
