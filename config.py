#!/usr/bin/env python3
import logging

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

