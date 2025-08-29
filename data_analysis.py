import pandas as pd
import random
from datetime import datetime, timedelta
from faker import Faker

def generate_synthetic_data(num_jobs=10, num_machines=5):
    """Generate synthetic data for factory scheduling"""
    fake = Faker()
    
    # Product types with their average required hours
    product_types = {
        "Smartphone": (3, 6),      # (min_hours, max_hours)
        "Laptop": (5, 10),
        "Tablet": (2, 5),
        "Monitor": (3, 7),
        "Keyboard": (1, 3),
        "Headphones": (2, 4),
        "Printer": (4, 8)
    }
    
    # Job types (operations) with their descriptions
    job_types = [
        "Assembly", 
        "Welding", 
        "Painting", 
        "Precision Cutting", 
        "Quality Testing", 
        "Circuit Soldering",
        "Component Installation",
        "Polishing",
        "Packaging",
        "Engraving"
    ]
    
    # Machine types with their specialties
    machine_types = [
        "Precision Assembler",
        "Advanced Welder",
        "Robotic Painter",
        "CNC Cutter",
        "Quality Control Station",
        "Soldering Station",
        "Installation Robot",
        "Surface Finisher",
        "Packaging System",
        "Laser Engraver"
    ]
    
    # Generate job data
    jobs = []
    for i in range(1, num_jobs + 1):
        product = random.choice(list(product_types.keys()))
        min_hours, max_hours = product_types[product]
        
        # Select a job type instead of generating a generic ID
        job_type = random.choice(job_types)
        job_id = f"{job_type}-{i}"
        
        # Random required hours based on product type
        required_hours = random.randint(min_hours, max_hours)
        
        # Random priority (1-3, where 3 is highest)
        priority = random.randint(1, 3)
        
        # Random deadline between 1-14 days from now
        days_ahead = random.randint(1, 14)
        deadline = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Add a description for each job
        description = f"{job_type} of {product} with {fake.word()} finish and {fake.word()} specifications"
        
        jobs.append({
            "job_id": job_id,
            "job_type": job_type,  # Make sure this column is included
            "product_type": product,
            "required_hours": required_hours,
            "priority": priority,
            "deadline": deadline,
            "description": description
        })
    
    # Generate machine data
    machines = []
    for i in range(1, num_machines + 1):
        # Select a machine type instead of generic ID
        machine_type = random.choice(machine_types)
        machine_id = f"{machine_type}-{i}"
        
        # Random available hours (4-12)
        available_hours = random.randint(4, 12)
        
        # Status (80% chance of being online)
        status = "online" if random.random() < 0.8 else "offline"
        
        # Generate machine skills
        skills_list = ["precision cutting", "assembly", "welding", "painting", 
                     "polishing", "testing", "packaging", "molding", "stamping"]
        skills_count = random.randint(2, 5)
        skills = f"Capabilities: {', '.join(random.sample(skills_list, skills_count))}"
        
        machines.append({
            "machine_id": machine_id,
            "machine_type": machine_type,
            "available_hours": available_hours,
            "status": status,
            "skills": skills
        })
    
    return pd.DataFrame(jobs), pd.DataFrame(machines)

def get_job_data():
    """Return job data for the agent (for compatibility with agent.py)"""
    jobs, _ = generate_synthetic_data(num_jobs=15)
    return jobs

def get_machine_data():
    """Return machine data for the agent (for compatibility with agent.py)"""
    _, machines = generate_synthetic_data(num_machines=5)
    return machines