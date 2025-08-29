import requests
import json

# Test the chatbot API
def test_chatbot_api():
    print("Testing chatbot API...")
    
    url = "http://127.0.0.1:5001/api/chatbot"
    headers = {"Content-Type": "application/json"}
    
    # Test with different queries
    test_queries = [
        "What is the machine status?",
        "Show me unscheduled jobs",
        "What's the overall factory status?",
        "Any urgent deadlines?"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        try:
            data = {"query": query}
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                print("Response JSON:")
                response_data = response.json()
                print(json.dumps(response_data, indent=2))
            else:
                print(f"Error response: {response.text}")
                
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    test_chatbot_api()