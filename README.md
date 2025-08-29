# AI-Powered Factory Scheduler

An intelligent factory scheduling system that uses BERT embeddings for semantic matching between jobs and machines.

## Features

- Semantic job-machine matching using BERT language model
- Priority-based job scheduling with deadline constraints
- Machine failure handling and job rescheduling
- Interactive web dashboard for schedule visualization
- API endpoints for integration with other systems

## Technologies

- Python 3.x
- Flask
- BERT (Transformers library)
- Pandas & NumPy
- Plotly for visualizations
- HTML/CSS/JavaScript frontend

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python3 app.py`

## Usage

Access the web interface at http://localhost:5001 after starting the application.

## API Documentation

The system provides the following API endpoints:
- `/api/getJobs` - Get all current jobs
- `/api/getMachineStatus` - Get machine status
- `/api/getSchedule` - Get current schedule
- `/api/scheduleJob` - Schedule a specific job
- `/api/recommendJobs` - Get job recommendations