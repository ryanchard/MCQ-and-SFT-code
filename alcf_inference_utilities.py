import requests

def get_names_of_alcf_chat_models():
    # Define the URL and headers for ALCF Inference Service list-endpoints
    url = "https://data-portal-dev.cels.anl.gov/resource_server/list-endpoints"
    headers = {
        "Authorization": f"Bearer {alcf_access_token}"
    }
    
    # Make the GET request
    response = requests.get(url, headers=headers)
    
    # Check the response
    if response.status_code == 200:
        status = response.json()
    else:
        print("Error:", response.status_code, response.text)
        exit(1)
    
    alcf_chat_models = status['clusters']['sophia']['frameworks']['vllm']['models']
    return alcf_chat_models
