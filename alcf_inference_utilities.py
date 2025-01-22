import requests

def get_names_of_alcf_chat_models(alcf_access_token):
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


def get_alcf_inference_service_model_queues(alcf_access_token):
    # Define the URL and headers
    url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/jobs"
    headers = { "Authorization": f"Bearer {alcf_access_token}" }
    
    # Make the GET request
    response = requests.get(url, headers=headers)
    
    # Check the response
    if response.status_code == 200:
        status = response.json()
    else:
        print("Error:", response.status_code, response.text)
        exit(1)
    
    running_models = status['running']
    running_model_list = []
    for model in running_models:
        running_model_list += model['Models Served'].split(',')
        if model['Model Status'] != 'running':
            print(f'SURPRISE: Expected *running* but got {model["Model Status"]}')

    queued_models = status['queued']
    queued_model_list = []
    for model in queued_models:
        queued_model_list += model['Models Served'].split(',')
        if model['Model Status'] != 'starting':
            print(f'SURPRISE: Expected *starting* but got {model["Model Status"]}')

    return(running_model_list, queued_model_list)
