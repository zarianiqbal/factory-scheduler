"""
Factory Scheduler Analytics Module
This module provides data analysis and visualization for the Factory Scheduler.
It can be run separately from the web app (app.py) to analyze scheduling data.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from data_analysis import generate_synthetic_data

# Make sure main.py can access the same data files as app.py
def ensure_data_exists():
    """Make sure data files exist, generating them if needed"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if data files exist
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
        return jobs_df, machines_df
    else:
        # Load existing data
        jobs_df = pd.read_csv('data/jobs.csv')
        machines_df = pd.read_csv('data/machines.csv')
        return jobs_df, machines_df

def perform_eda_on_scheduling_data(jobs_df, machines_df):
    """Perform simplified exploratory data analysis on scheduling data"""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    
    # -------- 1. DATA OVERVIEW DASHBOARD --------
    # Create a dashboard with key information (first visualization)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "table"}, {"type": "pie"}],
            [{"type": "bar", "colspan": 2}, None]
        ],
        subplot_titles=(
            "Sample Job Data", 
            "Machine Status Distribution", 
            "Product Type Distribution"
        ),
        column_widths=[0.6, 0.4],
        row_heights=[0.4, 0.6]
    )
    
    # Add a table with sample job data (without job_id column but with description)
    sample_jobs = jobs_df.head(5).drop(columns=['job_id'])
    
    # Ensure we have unique job_type/product_type combinations in the sample
    unique_combinations = jobs_df.drop_duplicates(['job_type', 'product_type']).head(5)
    if len(unique_combinations) >= 5:
        sample_jobs = unique_combinations.drop(columns=['job_id'])
    
    # Add a machine column with different machine types
    if 'assigned_machine' in sample_jobs.columns:
        sample_jobs = sample_jobs.drop(columns=['assigned_machine'])
    
    # Get some machine types from the machines dataframe
    machine_types = machines_df['machine_type'].unique()
    sample_machines = []
    
    # Assign a different machine to each row
    for i in range(len(sample_jobs)):
        if i < len(machine_types):
            sample_machines.append(machine_types[i])
        else:
            # In case we have more sample jobs than machine types
            sample_machines.append(f"Machine {i+1}")
    
    sample_jobs['machine'] = sample_machines
    
    # Rename description column to job_description for clarity
    if 'description' in sample_jobs.columns:
        sample_jobs = sample_jobs.rename(columns={'description': 'job_description'})
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(sample_jobs.columns),
                fill_color='royalblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[sample_jobs[col] for col in sample_jobs.columns],
                fill_color='lavender',
                align='left'
            )
        ),
        row=1, col=1
    )
    
    # Add pie chart for machine status
    status_counts = machines_df['status'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            textinfo='percent+label',
            hole=.3,
            marker_colors=['green', 'red']
        ),
        row=1, col=2
    )
    
    # Add bar chart for product type distribution
    product_counts = jobs_df['product_type'].value_counts()
    fig.add_trace(
        go.Bar(
            x=product_counts.index,
            y=product_counts.values,
            text=product_counts.values,
            textposition='auto',
            marker_color='royalblue'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title_text="Factory Scheduling Data Overview",
        height=800,
        width=1000,
        showlegend=True,
        margin=dict(b=150),  # Add bottom margin for annotation
        annotations=[
            dict(
                text="This dashboard provides an overview of factory scheduling data.<br>It shows sample job details, machine status distribution, and product types being manufactured.",
                align="center",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,  # Center horizontally
                y=-0.15,  # Position below the chart
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )
        ]
    )
    
    # Show the figure
    fig.show()
    
    # -------- 2. JOB PRIORITY ANALYSIS --------
    # Create a grouped bar chart for job priorities by job type
    job_priority_data = jobs_df.groupby(['job_type', 'priority']).size().reset_index(name='count')
    priority_labels = {1: "Low", 2: "Medium", 3: "High"}
    
    # Add a text column for priority
    job_priority_data['priority_label'] = job_priority_data['priority'].map(priority_labels)
    
    # Create grouped bar chart
    priority_fig = px.bar(
        job_priority_data,
        x='job_type',
        y='count',
        color='priority_label',
        title='Job Priority Distribution by Job Type',
        labels={'job_type': 'Job Type', 'count': 'Number of Jobs', 'priority_label': 'Priority'},
        color_discrete_map={'Low': 'lightblue', 'Medium': 'royalblue', 'High': 'darkred'}
    )
    priority_fig.update_layout(
        height=500, 
        width=900, 
        xaxis_tickangle=-45,
        showlegend=True,
        legend_title="Priority Level"
    )
    priority_fig.show()
    
    # -------- 3. MACHINE HOURS ANALYSIS --------
    # Use a single, more readable horizontal bar chart for machine hours
    online_machines = machines_df[machines_df['status'] == 'online']
    if not online_machines.empty:
        # Create a horizontal bar chart for machine hours
        fig = px.bar(
            online_machines,
            y='machine_id',
            x='available_hours',
            orientation='h',
            text='available_hours',
            color='available_hours',
            color_continuous_scale='Blues',
            title='Available Hours by Machine',
            labels={
                'machine_id': 'Machine',
                'available_hours': 'Available Hours'
            },
            height=400,
            width=800
        )
        
        # Add custom hover template
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Available Hours: %{x}<extra></extra>',
            textposition='outside'
        )
        
        # Add additional styling
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Available Hours',
            yaxis_title='Machine',
            xaxis={'range': [0, max(online_machines['available_hours']) * 1.1]},  # Add 10% padding
            coloraxis_colorbar=dict(
                title="Hours Available",
                thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=300,
                yanchor="top", y=1,
                ticks="outside"
            )
        )
        
        # Show the figure
        fig.show()
    
    # Print key insights with machine details
    print("\n=== KEY INSIGHTS ===")
    print(f"Total jobs: {len(jobs_df)}")
    print(f"Total machines: {len(machines_df)}")
    print(f"Online machines: {len(online_machines)}")
    
    # Print summary of each machine
    print("\nMachine Summary:")
    for _, machine in machines_df.iterrows():
        status_icon = "✓" if machine['status'] == 'online' else "✗"
        print(f"{status_icon} {machine['machine_id']}: {machine['available_hours']} hours available")
    
    print(f"\nTotal required hours: {jobs_df['required_hours'].sum()}")
    total_available = online_machines['available_hours'].sum() if not online_machines.empty else 0
    print(f"Total available hours: {total_available}")
    
    # Try to get current schedule data if available
    try:
        schedule_df = pd.read_csv('data/schedule.csv')
        if not schedule_df.empty:
            # Get scheduled vs unscheduled jobs
            scheduled_jobs = set(schedule_df['job_id'].unique())
            scheduled_count = len(scheduled_jobs)
            unscheduled_count = len(jobs_df) - scheduled_count
            
            print(f"\nScheduled jobs: {scheduled_count}")
            print(f"Unscheduled jobs: {unscheduled_count}")
            
            # Show distribution of jobs by machine
            if scheduled_count > 0:
                machine_job_counts = schedule_df['machine_id'].value_counts()
                print("\nJobs per machine:")
                for machine_id, count in machine_job_counts.items():
                    print(f"  {machine_id}: {count} jobs")
    except Exception as e:
        print("\nNo schedule data available yet.")

def analyze_job_distribution_by_deadline(jobs_df):
    """Analyze and visualize job distribution by deadline"""
    # Convert deadline to datetime if needed
    if 'deadline' in jobs_df.columns and not pd.api.types.is_datetime64_any_dtype(jobs_df['deadline']):
        jobs_df['deadline'] = pd.to_datetime(jobs_df['deadline'])
    
    # Count jobs by deadline
    jobs_df['deadline_date'] = jobs_df['deadline'].dt.date
    deadline_counts = jobs_df.groupby('deadline_date').size().reset_index(name='count')
    
    # Create a line chart showing job deadlines over time
    fig = px.line(
        deadline_counts, 
        x='deadline_date', 
        y='count',
        markers=True,
        title='Job Deadlines Distribution Over Time',
        labels={'deadline_date': 'Deadline Date', 'count': 'Number of Jobs Due'}
    )
    
    fig.update_layout(
        xaxis_title='Deadline Date',
        yaxis_title='Number of Jobs',
        height=400,
        width=800
    )
    
    fig.show()
    
    # Show deadline distribution by priority
    priority_deadline = jobs_df.groupby(['priority', pd.Grouper(key='deadline', freq='D')]).size().reset_index(name='count')
    
    # Map priority to labels
    priority_labels = {1: "Low", 2: "Medium", 3: "High"}
    priority_deadline['priority_label'] = priority_deadline['priority'].map(priority_labels)
    
    # Create a grouped bar chart
    fig = px.bar(
        priority_deadline,
        x='deadline',
        y='count',
        color='priority_label',
        title='Job Deadlines by Priority',
        labels={'deadline': 'Deadline Date', 'count': 'Number of Jobs', 'priority_label': 'Priority'},
        color_discrete_map={'Low': 'lightblue', 'Medium': 'royalblue', 'High': 'darkred'}
    )
    
    fig.update_layout(
        xaxis_title='Deadline Date',
        yaxis_title='Number of Jobs',
        height=400,
        width=800,
        bargap=0.2
    )
    
    fig.show()

def main():
    print("\n===== FACTORY SCHEDULER ANALYSIS TOOL =====")
    print("This tool provides data analysis and visualization")
    print("Running on port 5001 (web app uses port 5000)\n")
    print("NOTE: This analysis tool only visualizes data and does not perform scheduling")
    
    # Get data, either from existing files or generate new ones
    jobs_df, machines_df = ensure_data_exists()
    
    # Perform EDA on the data
    perform_eda_on_scheduling_data(jobs_df, machines_df)
    
    # Additional analysis for job deadlines
    analyze_job_distribution_by_deadline(jobs_df)
    
    print("\nData analysis complete!")
    print("\nNote: For scheduling functionality, use the web application with 'python app.py'")

if __name__ == "__main__":
    main()

