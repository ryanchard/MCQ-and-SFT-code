import requests

def get_names_of_alcf_chat_models(alcf_access_token):
    # Define the URL and headers for ALCF Inference Service list-endpoints
    url = "https://data-portal-dev.cels.anl.gov/resource_server/list-endpoints"
    headers = {
        "Authorization": f"Bearer {alcf_access_token}"
    }

    try:
        # Set a timeout so that the request doesn't hang indefinitely
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print("Could not connect. Check network connectivity and you must be local or connected via VPN", flush=True)
        exit(1)

    try:
        status = response.json()
    except ValueError:
        print("Error: Received an invalid JSON response.", flush=True)
        exit(1)

    alcf_chat_models = status['clusters']['sophia']['frameworks']['vllm']['models']
    return alcf_chat_models


def get_alcf_inference_service_model_queues(alcf_access_token):
    # Define the URL and headers
    url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/jobs"
    headers = { "Authorization": f"Bearer {alcf_access_token}" }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Could not connect. Check network connectivity and you must be local or connected via VPN", flush=True)
        exit(1)

    try:
        status = response.json()
    except ValueError:
        print("Error: Received an invalid JSON response.", flush=True)
        exit(1)

    running_models = status['running']
    running_model_list = []
    for model in running_models:
        running_model_list += model['Models Served'].split(',')
        if model['Model Status'] != 'running':
            print(f'SURPRISE: Expected *running* but got {model["Model Status"]}', flush=True)

    queued_models = status['queued']
    queued_model_list = []
    for model in queued_models:
        queued_model_list += model['Models Served'].split(',')
        if model['Model Status'] != 'starting':
            print(f'SURPRISE: Expected *starting* but got {model["Model Status"]}', flush=True)

    return (running_model_list, queued_model_list)

