import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from data_analysis import get_job_data, get_machine_data

# Load BERT model and tokenizer
print("Loading BERT model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

class FactorySchedulerBERTAgent:
    """
    Factory scheduler agent using BERT for semantic matching.
    
    System prompt:
    You are a factory scheduler agent. You receive job requests with deadlines, priority, 
    and duration. Assign each job to a machine with enough time available. Maximize priority 
    and ensure deadlines are met. Reassign if a machine goes offline. Justify your decisions.
    """
    
    def __init__(self, jobs_df=None, machines_df=None):
        """Initialize the agent with jobs and machines data"""
        # Load data if not provided
        self.jobs = jobs_df if jobs_df is not None else get_job_data()
        self.machines = machines_df if machines_df is not None else get_machine_data()
        
        # Convert deadline to datetime if needed
        if 'deadline' in self.jobs.columns and not pd.api.types.is_datetime64_any_dtype(self.jobs['deadline']):
            self.jobs['deadline'] = pd.to_datetime(self.jobs['deadline'])
        
        # Track machine capacity
        self.machine_capacity = {
            row['machine_id']: row['available_hours'] 
            for _, row in self.machines.iterrows() 
            if row['status'] == 'online'
        }
        
        # Keep track of job embeddings and machine embeddings
        self.job_embeddings = {}
        self.machine_embeddings = {}

    def generate_embeddings(self):
        """Generate BERT embeddings for job and machine descriptions"""
        # Generate job embeddings
        for _, job in self.jobs.iterrows():
            # Create a meaningful description for embedding
            if 'description' in job:
                job_desc = job['description']
            else:
                job_desc = f"{job['job_type']} of {job['product_type']} with priority {job['priority']}"
            
            # Get embedding
            with torch.no_grad():
                inputs = tokenizer(job_desc, return_tensors='pt', padding=True, truncation=True, max_length=128)
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                self.job_embeddings[job['job_id']] = embedding
        
        # Generate machine embeddings
        for _, machine in self.machines.iterrows():
            # Use skills or machine type for embedding
            if 'skills' in machine:
                machine_desc = f"{machine['machine_type']} with {machine['skills']}"
            else:
                machine_desc = machine['machine_type']
            
            # Get embedding
            with torch.no_grad():
                inputs = tokenizer(machine_desc, return_tensors='pt', padding=True, truncation=True, max_length=128)
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                self.machine_embeddings[machine['machine_id']] = embedding
                
        print(f"Generated embeddings for {len(self.job_embeddings)} jobs and {len(self.machine_embeddings)} machines")
                
    def calculate_similarity(self, job_id, machine_id):
        """Calculate similarity between a job and machine"""
        if job_id not in self.job_embeddings or machine_id not in self.machine_embeddings:
            return 0.0
        
        job_embedding = self.job_embeddings[job_id]
        machine_embedding = self.machine_embeddings[machine_id]
        
        # Reshape embeddings
        job_embedding = job_embedding.reshape(1, -1)
        machine_embedding = machine_embedding.reshape(1, -1)
        
        similarity = cosine_similarity(job_embedding, machine_embedding)[0][0]
        return similarity
        
    def schedule_jobs(self):
        """Schedule jobs to machines based on semantic similarity and constraints"""
        print("Generating embeddings for semantic matching...")
        self.generate_embeddings()
        
        # Sort jobs by priority and deadline
        sorted_jobs = self.jobs.sort_values(['priority', 'deadline'], ascending=[False, True])
        
        matches = []
        
        for _, job in sorted_jobs.iterrows():
            job_id = job['job_id']
            best_machine = None
            best_similarity = -1
            required_hours = job['required_hours']
            
            # Find the best machine with enough capacity
            for machine_id, capacity in self.machine_capacity.items():
                # Skip machines without enough capacity
                if capacity < required_hours:
                    continue
                    
                # Calculate semantic similarity
                similarity = self.calculate_similarity(job_id, machine_id)
                
                # Update best match if this is better
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_machine = machine_id
            
            # If we found a match, update capacity
            if best_machine:
                self.machine_capacity[best_machine] -= required_hours
                
            # Record the match (or lack thereof)
            matches.append({
                'job_id': job_id,
                'machine_id': best_machine,
                'similarity': best_similarity if best_machine else 0
            })
            
        return matches

def match_jobs_to_machines(jobs_df=None, machines_df=None):
    """Function to match jobs to machines using BERT-based semantic matching"""
    # Get data if not provided
    if jobs_df is None:
        jobs_df = get_job_data()
    if machines_df is None:
        machines_df = get_machine_data()
        
    agent = FactorySchedulerBERTAgent(jobs_df, machines_df)
    return agent.schedule_jobs()

if __name__ == "__main__":
    # Simple test code
    print("Initializing factory scheduler agent...")
    agent = FactorySchedulerBERTAgent()
    print("Agent initialized and ready for scheduling.")
