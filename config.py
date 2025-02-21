#!/usr/bin/env python3
import logging
import os
import yaml

# Define quiet mode flag (default is False)
quietMode = False

# Define default model (if not specified on command line with -m)
defaultModel = 'openai:gpt-4o'

# Configure logging
logger = logging.getLogger("MCQGenerator")  # Unique logger name
logger.setLevel(logging.INFO)  # Default logging level

# Create a console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Function to enable quiet mode
def set_quiet_mode(enable: bool):
    global quietMode
    quietMode = enable
    if quietMode:
        logger.setLevel(logging.WARNING)  # Suppress INFO messages
    else:
        logger.setLevel(logging.INFO)  # Show INFO messages

def get_quiet_mode() -> bool:
    return quietMode

# Load the prompts from prompt.yml
def load_prompts(file_path='prompts.yml'):
    """
    Safely load prompt definitions from a YAML file.
    """
    if not os.path.exists(file_path):
        logger.error(f"Prompts file '{file_path}' not found.")
        raise FileNotFoundError(f"Prompts file '{file_path}' not found.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file '{file_path}': {exc}")
        raise

# Load prompt messages from prompts.yml
_prompts = load_prompts()

# Expose prompt variables from the YAML file
system_message      = _prompts.get('system_message', "")
user_message        = _prompts.get('user_message', "")
system_message_2    = _prompts.get('system_message_2', "")
user_message_2      = _prompts.get('user_message_2', "")
system_message_3    = _prompts.get('system_message_3', "")
user_message_3      = _prompts.get('user_message_3', "")

