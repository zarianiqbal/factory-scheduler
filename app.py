from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import socket
import re
from data_analysis import generate_synthetic_data, get_job_data, get_machine_data
from core.agent import FactorySchedulerBERTAgent

app = Flask(__name__, static_folder='static', template_folder='templates')

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Initialize data files if they don't exist
def initialize_data_files():
    if not os.path.exists('data/jobs.csv') or not os.path.exists('data/machines.csv'):
        print("Generating initial data...")
        jobs_df, machines_df = generate_synthetic_data(num_jobs=15, num_machines=5)
        jobs_df.to_csv('data/jobs.csv', index=False)
        machines_df.to_csv('data/machines.csv', index=False)
        
        # Initialize empty schedule
        schedule_df = pd.DataFrame(columns=['job_id', 'machine_id', 'start_date', 'end_date', 'status'])
        schedule_df.to_csv('data/schedule.csv', index=False)
        
        # Initialize empty alerts
        alerts_df = pd.DataFrame(columns=['timestamp', 'type', 'message', 'severity', 'resolved'])
        alerts_df.to_csv('data/alerts.csv', index=False)
        print("Data initialization complete.")

def load_data():
    # Make sure files exist
    initialize_data_files()
    
    jobs_df = pd.read_csv('data/jobs.csv')
    machines_df = pd.read_csv('data/machines.csv')
    
    # Handle missing schedule or alerts files
    try:
        schedule_df = pd.read_csv('data/schedule.csv')
    except FileNotFoundError:
        schedule_df = pd.DataFrame(columns=['job_id', 'machine_id', 'start_date', 'end_date', 'status'])
        schedule_df.to_csv('data/schedule.csv', index=False)
    
    try:
        alerts_df = pd.read_csv('data/alerts.csv')
    except FileNotFoundError:
        alerts_df = pd.DataFrame(columns=['timestamp', 'type', 'message', 'severity', 'resolved'])
        alerts_df.to_csv('data/alerts.csv', index=False)
    
    # Convert date columns
    if 'deadline' in jobs_df.columns:
        jobs_df['deadline'] = pd.to_datetime(jobs_df['deadline'])
        
    if 'start_date' in schedule_df.columns and len(schedule_df) > 0:
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
        
    if 'timestamp' in alerts_df.columns and len(alerts_df) > 0:
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    
    return jobs_df, machines_df, schedule_df, alerts_df

def save_data(df, filename):
    df.to_csv(f'data/{filename}.csv', index=False)

def generate_alert(alert_type, message, severity='medium'):
    jobs_df, machines_df, schedule_df, alerts_df = load_data()
    
    new_alert = pd.DataFrame([{
        'timestamp': datetime.now(),
        'type': alert_type,
        'message': message,
        'severity': severity,
        'resolved': False
    }])
    
    alerts_df = pd.concat([alerts_df, new_alert], ignore_index=True)
    save_data(alerts_df, 'alerts')
    return True

# Add a new function to update machine status after recommendations
def update_machine_status_with_recommendations(recommendations):
    """Updates machine status based on recommendations"""
    jobs_df, machines_df, schedule_df, _ = load_data()
    
    # Store original remaining hours to avoid reloading data
    machine_hours = {}
    for _, machine in machines_df.iterrows():
        if machine['status'] == 'online':
            machine_hours[machine['machine_id']] = machine['available_hours']
    
    # Calculate current usage
    if not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            machine_id = row['machine_id']
            if machine_id in machine_hours:
                job = jobs_df[jobs_df['job_id'] == row['job_id']]
                if not job.empty:
                    machine_hours[machine_id] -= job.iloc[0]['required_hours']
    
    # Apply recommendations to update machine hours
    for rec in recommendations:
        machine_id = rec.get('machine_id')
        job_id = rec.get('job_id')
        
        if machine_id and job_id and machine_id in machine_hours:
            job = jobs_df[jobs_df['job_id'] == job_id]
            if not job.empty:
                machine_hours[machine_id] -= job.iloc[0]['required_hours']
    
    # Update machines DataFrame with new usage info
    for idx, machine in machines_df.iterrows():
        machine_id = machine['machine_id']
        if machine_id in machine_hours:
            total_hours = machine['available_hours']
            remaining_hours = max(0, machine_hours[machine_id])
            used_hours = total_hours - remaining_hours
            
            machines_df.at[idx, 'used_hours'] = used_hours
            machines_df.at[idx, 'remaining_hours'] = remaining_hours
            machines_df.at[idx, 'utilization'] = (used_hours / total_hours * 100) if total_hours > 0 else 0
    
    return machines_df

# Update the clean_nan_values function to be more robust
def clean_nan_values(obj):
    """Recursively clean NaN values from an object to make it JSON serializable"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and (pd.isna(obj) or str(obj) == 'nan' or str(obj) == 'NaN'):
        return 0  # Replace NaN with 0
    else:
        return obj

# Update the ensure_json_serializable function to handle numpy types
def ensure_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return ensure_json_serializable(obj.to_dict())
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif pd.isna(obj):
        return None
    else:
        return obj

# API routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/api/getJobs')
def get_jobs():
    jobs_df, _, schedule_df, _ = load_data()
    
    # Add scheduled status to jobs
    if not schedule_df.empty:
        scheduled_jobs = set(schedule_df['job_id'].unique())
        jobs_df['scheduled'] = jobs_df['job_id'].apply(lambda x: x in scheduled_jobs)
    else:
        jobs_df['scheduled'] = False
        
    # Convert datetime to string for JSON serialization
    if 'deadline' in jobs_df.columns:
        jobs_df['deadline'] = jobs_df['deadline'].dt.strftime('%Y-%m-%d')
    
    return jsonify(jobs_df.to_dict(orient='records'))

@app.route('/api/getMachineStatus')
def get_machine_status():
    _, machines_df, schedule_df, _ = load_data()
    
    # Calculate machine utilization
    if not schedule_df.empty and not machines_df.empty:
        # Get jobs currently assigned to each machine
        current_assignments = schedule_df[schedule_df['status'] == 'scheduled']
        
        # Calculate hours used per machine
        machine_usage = {}
        for _, row in current_assignments.iterrows():
            machine_id = row['machine_id']
            if machine_id not in machine_usage:
                machine_usage[machine_id] = 0
                
            # Get job details
            jobs_df, _, _, _ = load_data()
            job = jobs_df[jobs_df['job_id'] == row['job_id']]
            if not job.empty:
                machine_usage[machine_id] += job.iloc[0]['required_hours']
        
        # Update machines with utilization info
        for idx, machine in machines_df.iterrows():
            machine_id = machine['machine_id']
            if machine_id in machine_usage:
                used_hours = machine_usage[machine_id]
                total_hours = machine['available_hours']
                remaining_hours = max(0, total_hours - used_hours)
                
                machines_df.at[idx, 'used_hours'] = used_hours
                machines_df.at[idx, 'remaining_hours'] = remaining_hours
                machines_df.at[idx, 'utilization'] = (used_hours / total_hours * 100) if total_hours > 0 else 0
            else:
                machines_df.at[idx, 'used_hours'] = 0
                machines_df.at[idx, 'remaining_hours'] = machine['available_hours']
                machines_df.at[idx, 'utilization'] = 0
    
    return jsonify(machines_df.to_dict(orient='records'))

@app.route('/api/getSchedule')
def get_schedule():
    jobs_df, _, schedule_df, _ = load_data()
    
    if schedule_df.empty:
        return jsonify([])
    
    # Convert datetime to string for JSON
    schedule_df['start_date'] = schedule_df['start_date'].dt.strftime('%Y-%m-%d')
    schedule_df['end_date'] = schedule_df['end_date'].dt.strftime('%Y-%m-%d')
    
    # Enrich schedule with job details
    enriched_schedule = []
    for _, row in schedule_df.iterrows():
        schedule_item = row.to_dict()
        job = jobs_df[jobs_df['job_id'] == row['job_id']]
        
        if not job.empty:
            job_info = job.iloc[0].to_dict()
            schedule_item.update({
                'product_type': job_info.get('product_type', ''),
                'job_type': job_info.get('job_type', ''),
                'priority': job_info.get('priority', 0),
                'required_hours': job_info.get('required_hours', 0)
            })
            
        enriched_schedule.append(schedule_item)
    
    return jsonify(enriched_schedule)

@app.route('/api/getAlerts')
def get_alerts():
    _, _, _, alerts_df = load_data()
    
    if 'timestamp' in alerts_df.columns and len(alerts_df) > 0:
        alerts_df['timestamp'] = alerts_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Sort by timestamp (newest first) and limit to 100
    if not alerts_df.empty:
        alerts_df = alerts_df.sort_values('timestamp', ascending=False).head(100)
    
    return jsonify(alerts_df.to_dict(orient='records'))

@app.route('/api/scheduleJob', methods=['POST'])
def schedule_job():
    try:
        data = request.json
        job_id = data.get('job_id')
        machine_id = data.get('machine_id')
        
        if not job_id or not machine_id:
            return jsonify({'success': False, 'message': 'Job ID and Machine ID are required'}), 400
        
        jobs_df, machines_df, schedule_df, _ = load_data()
        
        # Check if job exists
        job = jobs_df[jobs_df['job_id'] == job_id]
        if job.empty:
            return jsonify({'success': False, 'message': f'Job {job_id} not found'}), 404
        
        # Check if machine exists and is online
        machine = machines_df[machines_df['machine_id'] == machine_id]
        if machine.empty:
            return jsonify({'success': False, 'message': f'Machine {machine_id} not found'}), 404
        
        if machine.iloc[0]['status'] != 'online':
            return jsonify({'success': False, 'message': f'Machine {machine_id} is offline'}), 400
        
        # Check if job is already scheduled
        if not schedule_df.empty and job_id in schedule_df['job_id'].values:
            return jsonify({'success': False, 'message': f'Job {job_id} is already scheduled'}), 400
        
        # Check if machine has enough capacity
        # Convert required_hours to a Python native int to fix the numpy.int64 error
        required_hours = int(job.iloc[0]['required_hours'])
        
        # Get current machine usage
        current_usage = 0
        if not schedule_df.empty:
            machine_jobs = schedule_df[schedule_df['machine_id'] == machine_id]
            for _, row in machine_jobs.iterrows():
                job_info = jobs_df[jobs_df['job_id'] == row['job_id']]
                if not job_info.empty:
                    current_usage += int(job_info.iloc[0]['required_hours'])  # Convert to native int
        
        available_hours = int(machine.iloc[0]['available_hours'])  # Convert to native int
        if current_usage + required_hours > available_hours:
            return jsonify({
                'success': False, 
                'message': f'Machine {machine_id} does not have enough capacity (needs {required_hours} hours, has {available_hours - current_usage} hours available)'
            }), 400
        
        # Schedule the job
        start_date = datetime.now()
        # Convert required_hours to Python native int to avoid numpy.int64 error with timedelta
        end_date = start_date + timedelta(hours=required_hours)
        
        new_schedule = pd.DataFrame([{
            'job_id': job_id,
            'machine_id': machine_id,
            'start_date': start_date,
            'end_date': end_date,
            'status': 'scheduled'
        }])
        
        schedule_df = pd.concat([schedule_df, new_schedule], ignore_index=True)
        save_data(schedule_df, 'schedule')
        
        # Generate alert
        generate_alert('job_scheduled', f'Job {job_id} scheduled on machine {machine_id}', 'info')
        
        return jsonify({'success': True, 'message': f'Job {job_id} scheduled successfully on machine {machine_id}'})
    
    except Exception as e:
        print(f"Error in schedule_job API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/api/recommendJobs')
def recommend_jobs():
    jobs_df, machines_df, schedule_df, _ = load_data()
    
    # Get unscheduled jobs
    if not schedule_df.empty:
        scheduled_jobs = set(schedule_df['job_id'].unique())
        unscheduled_jobs = jobs_df[~jobs_df['job_id'].isin(scheduled_jobs)]
    else:
        unscheduled_jobs = jobs_df
    
    if unscheduled_jobs.empty:
        return jsonify({'recommendations': [], 'message': 'No unscheduled jobs found'})
    
    # Initialize the BERT agent to get recommendations
    try:
        agent = FactorySchedulerBERTAgent(unscheduled_jobs, machines_df)
        matches = agent.schedule_jobs()
        
        # Format recommendations
        recommendations = []
        for match in matches:
            job_id = match['job_id']
            machine_id = match['machine_id']
            similarity = match['similarity']
            
            if machine_id is None:
                continue
                
            # Make sure we're using the correct case for job_id
            # by looking it up in the original dataframe
            job_info = unscheduled_jobs[unscheduled_jobs['job_id'].str.lower() == job_id.lower()]
            if job_info.empty:
                continue
                
            # Use the correct case for job_id from the dataframe
            actual_job_id = job_info.iloc[0]['job_id']
            
            # Similarly ensure correct case for machine_id
            machine_info = machines_df[machines_df['machine_id'].str.lower() == machine_id.lower()]
            actual_machine_id = machine_id
            if not machine_info.empty:
                actual_machine_id = machine_info.iloc[0]['machine_id']
                
            recommendations.append({
                'job_id': actual_job_id,  # Use the correct case
                'machine_id': actual_machine_id,  # Use the correct case
                'similarity_score': similarity,
                'priority': int(job_info.iloc[0]['priority']),
                'required_hours': int(job_info.iloc[0]['required_hours']),
                'deadline': job_info.iloc[0]['deadline'].strftime('%Y-%m-%d') if isinstance(job_info.iloc[0]['deadline'], pd.Timestamp) else job_info.iloc[0]['deadline'],
                'product_type': job_info.iloc[0]['product_type'],
                'job_type': job_info.iloc[0]['job_type']
            })
        
        # Update machine status based on recommendations
        updated_machines = update_machine_status_with_recommendations(recommendations)
        
        # Generate an alert about the recommendations
        if recommendations:
            generate_alert(
                'recommendations_generated', 
                f'Generated {len(recommendations)} job-machine recommendations', 
                'info'
            )
        
        # Clean any NaN values to ensure valid JSON
        recommendations = clean_nan_values(recommendations)
        machines_data = clean_nan_values(updated_machines.to_dict(orient='records'))
        
        # Use safer JSON serialization
        import json
        class NaNSafeJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float) and (pd.isna(obj) or str(obj) == 'nan' or str(obj) == 'NaN'):
                    return 0
                return super().default(obj)
        
        # Create a custom response with safe JSON handling
        response_data = {
            'recommendations': recommendations,
            'message': f'Generated {len(recommendations)} recommendations',
            'machines': machines_data
        }
        
        # Return a manually JSON-encoded response
        return app.response_class(
            response=json.dumps(response_data, cls=NaNSafeJSONEncoder),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        # Fallback to simple priority-based recommendations
        print(f"Error using BERT agent: {str(e)}")
        
        # Sort by priority and deadline
        recommendations = []
        sorted_jobs = unscheduled_jobs.sort_values(['priority', 'deadline'], ascending=[False, True])
        
        for _, job in sorted_jobs.iterrows():
            # Find available machines
            available_machines = machines_df[machines_df['status'] == 'online']
            
            for _, machine in available_machines.iterrows():
                recommendations.append({
                    'job_id': job['job_id'],
                    'machine_id': machine['machine_id'],
                    'similarity_score': 0.5,  # Default score
                    'priority': int(job['priority']),
                    'required_hours': int(job['required_hours']),
                    'deadline': job['deadline'].strftime('%Y-%m-%d') if isinstance(job['deadline'], pd.Timestamp) else job['deadline'],
                    'product_type': job['product_type'],
                    'job_type': job['job_type']
                })
                break  # Just recommend one machine per job
        
        # Update machine status based on recommendations
        updated_machines = update_machine_status_with_recommendations(recommendations)
        
        # Clean any NaN values to ensure valid JSON
        recommendations = clean_nan_values(recommendations)
        machines_data = clean_nan_values(updated_machines.to_dict(orient='records'))
        
        # Use safer JSON serialization
        import json
        class NaNSafeJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float) and (pd.isna(obj) or str(obj) == 'nan' or str(obj) == 'NaN'):
                    return 0
                return super().default(obj)
        
        # Create a custom response with safe JSON handling
        response_data = {
            'recommendations': recommendations,
            'message': f'Generated {len(recommendations)} recommendations (fallback mode)',
            'machines': machines_data
        }
        
        # Return a manually JSON-encoded response
        return app.response_class(
            response=json.dumps(response_data, cls=NaNSafeJSONEncoder),
            status=200,
            mimetype='application/json'
        )

@app.route('/api/getSystemStatus')
def get_system_status():
    jobs_df, machines_df, schedule_df, alerts_df = load_data()
    
    # Calculate key metrics
    total_jobs = len(jobs_df)
    scheduled_jobs = len(schedule_df['job_id'].unique()) if not schedule_df.empty else 0
    unscheduled_jobs = total_jobs - scheduled_jobs
    
    online_machines = len(machines_df[machines_df['status'] == 'online'])
    offline_machines = len(machines_df[machines_df['status'] == 'offline'])
    
    # Calculate deadline risks
    deadline_risks = []
    if not schedule_df.empty:
        for _, job in jobs_df.iterrows():
            if job['job_id'] in schedule_df['job_id'].values:
                # Job is scheduled, check if it might miss deadline
                job_schedule = schedule_df[schedule_df['job_id'] == job['job_id']].iloc[0]
                end_date = pd.to_datetime(job_schedule['end_date'])
                deadline = pd.to_datetime(job['deadline'])
                
                # If end date is after deadline or within 1 day of deadline
                if end_date > deadline or (deadline - end_date).days <= 1:
                    deadline_risks.append({
                        'job_id': job['job_id'],
                        'deadline': deadline.strftime('%Y-%m-%d'),
                        'expected_completion': end_date.strftime('%Y-%m-%d'),
                        'days_remaining': int((deadline - datetime.now()).days),
                        'product_type': job['product_type'],
                        'priority': int(job['priority'])
                    })
    
    # Calculate machine overloads
    machine_overloads = []
    for _, machine in machines_df.iterrows():
        if machine['status'] == 'offline':
            continue
            
        machine_id = machine['machine_id']
        total_hours = machine['available_hours']
        
        # Calculate used hours
        used_hours = 0
        if not schedule_df.empty:
            for _, schedule in schedule_df[schedule_df['machine_id'] == machine_id].iterrows():
                job = jobs_df[jobs_df['job_id'] == schedule['job_id']]
                if not job.empty:
                    used_hours += job.iloc[0]['required_hours']
        
        # If utilization is over 90%, consider it at risk of overload
        utilization = (used_hours / total_hours * 100) if total_hours > 0 else 0
        if utilization > 90:
            machine_overloads.append({
                'machine_id': machine_id,
                'utilization': float(utilization),  # Convert to float explicitly
                'used_hours': int(used_hours),      # Convert to int explicitly
                'total_hours': int(total_hours),    # Convert to int explicitly
                'remaining_hours': int(total_hours - used_hours)  # Convert to int explicitly
            })
    
    # Get recent alerts
    recent_alerts = []
    if not alerts_df.empty:
        recent_alerts = alerts_df.sort_values('timestamp', ascending=False).head(5).to_dict(orient='records')
        for alert in recent_alerts:
            if isinstance(alert['timestamp'], pd.Timestamp):
                alert['timestamp'] = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    
    # Create response data ensuring all values are JSON serializable
    response_data = {
        'total_jobs': int(total_jobs),
        'scheduled_jobs': int(scheduled_jobs),
        'unscheduled_jobs': int(unscheduled_jobs),
        'online_machines': int(online_machines),
        'offline_machines': int(offline_machines),
        'deadline_risks': deadline_risks,
        'machine_overloads': machine_overloads,
        'recent_alerts': recent_alerts
    }
    
    # Use the class to handle JSON serialization
    return jsonify(clean_nan_values(response_data))

@app.route('/api/chatbot', methods=['POST'])
def process_chatbot_query():
    try:
        query = request.json.get('query', '').lower()
        
        if not query:
            return jsonify({'response': 'Please provide a query'})
        
        print(f"Received chatbot query: {query}")
        
        # Ensure data files exist
        initialize_data_files()
        
        # Handle direct scheduling commands
        if 'schedule job' in query or 'assign job' in query:
            # Extract job ID and machine ID from the query
            job_pattern = r'job\s+([a-zA-Z0-9\-]+)'
            machine_pattern = r'machine\s+([a-zA-Z0-9\-]+)'
            
            job_match = re.search(job_pattern, query, re.IGNORECASE)
            machine_match = re.search(machine_pattern, query, re.IGNORECASE)
            
            if job_match and machine_match:
                # Get the job_id as it appears in the query, preserving case
                job_id_from_query = job_match.group(1)
                machine_id = machine_match.group(1)
                
                # Call the existing schedule API
                try:
                    # Validate the job and machine exist
                    jobs_df, machines_df, schedule_df, _ = load_data()
                    
                    # First try exact match
                    job = jobs_df[jobs_df['job_id'] == job_id_from_query]
                    
                    # If not found, try case-insensitive match
                    if job.empty:
                        # Get a mapping of lowercase job_ids to actual job_ids to handle case sensitivity
                        job_id_map = {j_id.lower(): j_id for j_id in jobs_df['job_id']}
                        actual_job_id = job_id_map.get(job_id_from_query.lower())
                        
                        if actual_job_id:
                            job = jobs_df[jobs_df['job_id'] == actual_job_id]
                            # Replace the query job_id with the actual job_id
                            job_id = actual_job_id
                        else:
                            return jsonify({'response': f"I couldn't find Job {job_id_from_query}. Please check the job ID and try again."})
                    else:
                        job_id = job_id_from_query
                    
                    # Similarly, try case-insensitive match for machine
                    machine = machines_df[machines_df['machine_id'] == machine_id]
                    if machine.empty:
                        machine_id_map = {m_id.lower(): m_id for m_id in machines_df['machine_id']}
                        actual_machine_id = machine_id_map.get(machine_id.lower())
                        
                        if actual_machine_id:
                            machine = machines_df[machines_df['machine_id'] == actual_machine_id]
                            machine_id = actual_machine_id
                        else:
                            return jsonify({'response': f"I couldn't find Machine {machine_id}. Please check the machine ID and try again."})
                    
                    # Check if the job is already scheduled
                    if not schedule_df.empty and job_id in schedule_df['job_id'].values:
                        return jsonify({'response': f"Job {job_id} is already scheduled."})
                    
                    # Check if machine is online
                    if machine.iloc[0]['status'] != 'online':
                        return jsonify({'response': f"Machine {machine_id} is currently offline. Please select an online machine."})
                    
                    # Check if machine has enough capacity
                    required_hours = job.iloc[0]['required_hours']
                    
                    # Get current machine usage
                    current_usage = 0
                    if not schedule_df.empty:
                        machine_jobs = schedule_df[schedule_df['machine_id'] == machine_id]
                        for _, row in machine_jobs.iterrows():
                            job_info = jobs_df[jobs_df['job_id'] == row['job_id']]
                            if not job_info.empty:
                                current_usage += job_info.iloc[0]['required_hours']
                    
                    available_hours = machine.iloc[0]['available_hours']
                    if current_usage + required_hours > available_hours:
                        return jsonify({'response': f"Machine {machine_id} does not have enough capacity for this job. It needs {required_hours} hours but only has {available_hours - current_usage} hours available."})
                    
                    # Schedule the job
                    start_date = datetime.now()
                    end_date = start_date + timedelta(hours=required_hours)
                    
                    new_schedule = pd.DataFrame([{
                        'job_id': job_id,
                        'machine_id': machine_id,
                        'start_date': start_date,
                        'end_date': end_date,
                        'status': 'scheduled'
                    }])
                    
                    schedule_df = pd.concat([schedule_df, new_schedule], ignore_index=True)
                    save_data(schedule_df, 'schedule')
                    
                    # Generate alert
                    generate_alert('job_scheduled', f'Job {job_id} scheduled on machine {machine_id} via chatbot', 'info')
                    
                    product_type = job.iloc[0]['product_type']
                    job_type = job.iloc[0]['job_type']
                    
                    return jsonify({
                        'response': f"✅ Success! I've scheduled Job {job_id} ({job_type} of {product_type}) on Machine {machine_id}.\n\n" +
                                   f"The job requires {required_hours} hours and has been added to the production schedule.",
                        'data_type': 'job_scheduled',
                        'job_id': job_id,
                        'machine_id': machine_id
                    })
                    
                except Exception as e:
                    print(f"Error scheduling job: {str(e)}")
                    return jsonify({'response': f"I encountered an error while scheduling: {str(e)}"})
            else:
                return jsonify({'response': "To schedule a job, please specify both the job ID and machine ID. For example: 'Schedule Job Assembly-2 on Machine Precision-1'"})
                
        # Process different types of queries
        elif 'unscheduled job' in query or 'jobs to schedule' in query:
            jobs_df, machines_df, schedule_df, _ = load_data()
            
            if not schedule_df.empty:
                scheduled_jobs = set(schedule_df['job_id'].unique())
                unscheduled = jobs_df[~jobs_df['job_id'].isin(scheduled_jobs)]
            else:
                unscheduled = jobs_df
                
            if unscheduled.empty:
                return jsonify({'response': 'All jobs have been scheduled! The factory is running at full capacity.'})
            
            # Sort by priority
            unscheduled = unscheduled.sort_values('priority', ascending=False)
            
            # Format response
            response = f"I found {len(unscheduled)} unscheduled jobs. Here are the top priority ones:\n\n"
            for i, (_, job) in enumerate(unscheduled.head(5).iterrows()):
                response += f"{i+1}. Job {job['job_id']}: {job['job_type']} of {job['product_type']}\n"
                response += f"   Priority: {job['priority']}, Required hours: {job['required_hours']}\n"
                response += f"   Deadline: {job['deadline'].strftime('%Y-%m-%d') if isinstance(job['deadline'], pd.Timestamp) else job['deadline']}\n\n"
                
            # NEW CODE: If the query requests recommendations, generate them
            if 'recommend' in query or 'match' in query or 'assign' in query:
                response += "Here are my machine recommendations for these jobs:\n\n"
                
                try:
                    # Use the BERT agent for recommendations
                    agent = FactorySchedulerBERTAgent(unscheduled, machines_df)
                    matches = agent.schedule_jobs()
                    
                    # Add recommendations to the response
                    for match in matches[:5]:  # Show top 5 matches
                        job_id = match['job_id']
                        machine_id = match['machine_id']
                        similarity = match['similarity']
                        
                        if machine_id is None:
                            continue
                            
                        job_info = unscheduled[unscheduled['job_id'] == job_id]
                        if job_info.empty:
                            continue
                            
                        job = job_info.iloc[0]
                        response += f"• Job {job_id} ({job['job_type']} of {job['product_type']}) → Machine {machine_id}\n"
                        response += f"  Match quality: {similarity:.2f}, Job priority: {job['priority']}\n"
                except Exception as e:
                    print(f"Error generating recommendations: {str(e)}")
                    # Fallback to a simple recommendation
                    response += "I couldn't generate detailed recommendations, but here's a simple match:\n\n"
                    
                    # Simple matching logic
                    online_machines = machines_df[machines_df['status'] == 'online']
                    for i, (_, job) in enumerate(unscheduled.head(5).iterrows()):
                        if i < len(online_machines):
                            machine = online_machines.iloc[i]
                            response += f"• Job {job['job_id']} → Machine {machine['machine_id']}\n"
                        else:
                            response += f"• Job {job['job_id']} → No available machine\n"
            else:
                response += "Would you like me to recommend machines for these jobs? Just ask for 'recommendations'."
            
            return jsonify({'response': response, 'data_type': 'unscheduled_jobs', 'count': len(unscheduled)})
        
        # Check for direct recommendation queries
        elif 'recommend' in query or 'match jobs' in query or 'assign jobs' in query:
            jobs_df, machines_df, schedule_df, _ = load_data()
            
            # Get unscheduled jobs
            if not schedule_df.empty:
                scheduled_jobs = set(schedule_df['job_id'].unique())
                unscheduled = jobs_df[~jobs_df['job_id'].isin(scheduled_jobs)]
            else:
                unscheduled = jobs_df
                
            if unscheduled.empty:
                return jsonify({'response': 'All jobs have been scheduled! There are no jobs to recommend machines for.'})
            
            # Generate recommendations
            response = "Here are my machine recommendations for the top priority jobs:\n\n"
            
            try:
                # Use the BERT agent for recommendations
                agent = FactorySchedulerBERTAgent(unscheduled, machines_df)
                matches = agent.schedule_jobs()
                
                # Add recommendations to the response
                for match in matches[:7]:  # Show top 7 matches (more than original response)
                    job_id = match['job_id']
                    machine_id = match['machine_id']
                    similarity = match['similarity']
                    
                    if machine_id is None:
                        continue
                        
                    job_info = unscheduled[unscheduled['job_id'] == job_id]
                    if job_info.empty:
                        continue
                        
                    job = job_info.iloc[0]
                    response += f"• Job {job_id} ({job['job_type']} of {job['product_type']}) → Machine {machine_id}\n"
                    response += f"  Match quality: {similarity:.2f}, Priority: {job['priority']}, Hours needed: {job['required_hours']}\n"
                    response += f"  Deadline: {job['deadline'].strftime('%Y-%m-%d') if isinstance(job['deadline'], pd.Timestamp) else job['deadline']}\n\n"
            except Exception as e:
                print(f"Error generating recommendations: {str(e)}")
                # Fallback to a simple recommendation
                response += "I couldn't generate detailed recommendations due to a technical issue, but here's a simple match based on priorities:\n\n"
                
                # Sort jobs by priority
                sorted_jobs = unscheduled.sort_values('priority', ascending=False)
                online_machines = machines_df[machines_df['status'] == 'online']
                
                for i, (_, job) in enumerate(sorted_jobs.head(5).iterrows()):
                    if i < len(online_machines):
                        machine = online_machines.iloc[i]
                        response += f"• Job {job['job_id']} ({job['job_type']} of {job['product_type']}) → Machine {machine['machine_id']}\n"
                        response += f"  Priority: {job['priority']}, Deadline: {job['deadline'].strftime('%Y-%m-%d') if isinstance(job['deadline'], pd.Timestamp) else job['deadline']}\n\n"
                    else:
                        response += f"• Job {job['job_id']} → No available machine\n\n"
            
            return jsonify({'response': response, 'data_type': 'job_recommendations'})
            
        elif 'machine status' in query or 'online machine' in query:
            _, machines_df, _, _ = load_data()
            
            online = machines_df[machines_df['status'] == 'online']
            offline = machines_df[machines_df['status'] == 'offline']
            
            response = f"Currently, {len(online)} machines are online and {len(offline)} are offline.\n\n"
            
            # Add details about online machines
            response += "Online machines:\n"
            for _, machine in online.iterrows():
                response += f"- {machine['machine_id']} ({machine['machine_type']}): {machine['available_hours']} hours available\n"
                
            if not offline.empty:
                response += "\nOffline machines:\n"
                for _, machine in offline.iterrows():
                    response += f"- {machine['machine_id']} ({machine['machine_type']})\n"
                    
            return jsonify({'response': response, 'data_type': 'machine_status', 'count': len(machines_df)})
            
        elif 'deadline' in query or 'urgent' in query:
            jobs_df, _, schedule_df, _ = load_data()
            
            # Find jobs with upcoming deadlines
            today = datetime.now()
            upcoming_deadlines = jobs_df[
                (jobs_df['deadline'] - today).dt.days <= 3
            ].sort_values('deadline')
            
            if upcoming_deadlines.empty:
                return jsonify({'response': 'There are no jobs with urgent deadlines (within 3 days).'})
                
            response = f"I found {len(upcoming_deadlines)} jobs with deadlines within the next 3 days:\n\n"
            
            for _, job in upcoming_deadlines.iterrows():
                days_left = (job['deadline'] - today).days
                days_text = f"{days_left} days left" if days_left > 0 else "DUE TODAY!"
                
                # Check if scheduled
                is_scheduled = False
                if not schedule_df.empty:
                    is_scheduled = job['job_id'] in schedule_df['job_id'].values
                    
                status = "✓ Scheduled" if is_scheduled else "⚠️ Not scheduled"
                
                response += f"• Job {job['job_id']}: {job['product_type']} ({days_text})\n"
                response += f"  Priority: {job['priority']}, Status: {status}\n"
                
            return jsonify({'response': response, 'data_type': 'upcoming_deadlines', 'count': len(upcoming_deadlines)})
        
        elif 'overall status' in query or 'factory status' in query or 'summary' in query:
            status_response = get_system_status()
            status_data = status_response.get_json()
            
            # Make sure all values are JSON serializable
            status_data = ensure_json_serializable(status_data)
            
            total_jobs = status_data['total_jobs']
            scheduled = status_data['scheduled_jobs']
            unscheduled = status_data['unscheduled_jobs']
            
            online = status_data['online_machines']
            offline = status_data['offline_machines']
            
            deadline_risks = status_data['deadline_risks']
            machine_overloads = status_data['machine_overloads']
            
            response = "📊 Factory Status Summary 📊\n\n"
            response += f"Jobs: {scheduled}/{total_jobs} scheduled ({unscheduled} waiting)\n"
            response += f"Machines: {online} online, {offline} offline\n\n"
            
            if deadline_risks:
                response += f"⚠️ {len(deadline_risks)} jobs at risk of missing deadlines\n"
                
            if machine_overloads:
                response += f"⚠️ {len(machine_overloads)} machines near capacity\n"
                
            if not deadline_risks and not machine_overloads:
                response += "✓ No urgent issues detected\n"
                
            return jsonify({'response': response, 'data_type': 'system_status'})
        
        else:
            return jsonify({
                'response': "I can help with factory scheduling tasks. Try asking about:\n\n" +
                            "• Unscheduled jobs\n" +
                            "• Machine status\n" +
                            "• Upcoming deadlines\n" +
                            "• Overall factory status\n" +
                            "• Recommend machines for jobs"
            })
            
    except Exception as e:
        print(f"Error in chatbot API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'response': f"I encountered an error: {str(e)}. Please try a different question."})

@app.route('/api/generateNewData')
def generate_new_data():
    """Generate fresh data for the factory scheduler"""
    try:
        # Generate new data
        print("Generating new factory data...")
        jobs_df, machines_df = generate_synthetic_data(num_jobs=15, num_machines=5)
        
        # Save the new data
        jobs_df.to_csv('data/jobs.csv', index=False)
        machines_df.to_csv('data/machines.csv', index=False)
        
        # Initialize empty schedule
        schedule_df = pd.DataFrame(columns=['job_id', 'machine_id', 'start_date', 'end_date', 'status'])
        schedule_df.to_csv('data/schedule.csv', index=False)
        
        # Generate a notification alert
        generate_alert(
            'data_refresh', 
            'New factory data has been generated', 
            'info'
        )
        
        return jsonify({
            'success': True, 
            'message': f'Successfully generated {len(jobs_df)} new jobs and {len(machines_df)} machines'
        })
        
    except Exception as e:
        print(f"Error generating new data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error generating new data: {str(e)}'}), 500

# Modify the port selection logic at the end of the file
if __name__ == '__main__':
    # Initialize data
    initialize_data_files()
    
    # Use port 5001 as default
    port = 5001
    print("\n======= Factory Scheduler Web Application =======")
    print("Starting Factory Scheduler application...")
    print(f"Access the dashboard at http://127.0.0.1:{port}")
    print(f"Access the chatbot at http://127.0.0.1:{port}/chatbot")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    except OSError:
        # If 5001 is busy, try 5002
        port = 5002
        print(f"\n⚠️ Port 5001 is already in use, trying port {port}...")
        print(f"Access the dashboard at http://127.0.0.1:{port}")
        print(f"Access the chatbot at http://127.0.0.1:{port}/chatbot")
        app.run(host='0.0.0.0', port=port, debug=True)