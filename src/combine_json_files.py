#!/usr/bin/env python

import json
from pathlib import Path
import argparse
import config

# Set up argument parsing
parser = argparse.ArgumentParser(description="Combine JSON files in a directory into a single JSON file.")
parser.add_argument('-i', '--input',  help='Directory containing input MCQ files',
                    default=config.mcq_dir)
parser.add_argument('-o', '--output', help='Output file for combined MCQs')

args = parser.parse_args()

# Get input directory and output file from arguments
input_directory = args.input
output_file = args.output

# Initialize an empty list to hold all JSON objects
combined_json_list = []

# Iterate through all JSON files in the directory
for json_file in Path(input_directory).glob("*.json"):
    with open(json_file, "r") as file:
        try:
            # Load JSON data from the file and extend the combined list
            data = json.load(file)
            if isinstance(data, list):
                combined_json_list.extend(data)
            else:
                print(f"Skipping {json_file}: Not a JSON list")
        except json.JSONDecodeError as e:
            print(f"Skipping {json_file}: Invalid JSON - {e}")

combined_json_list.sort(key=lambda x: (x.get("filenum", float('inf')), x.get("chunknum", float('inf'))))

# Write the combined list to the output file
with open(output_file, "w") as output:
    json.dump(combined_json_list, output, indent=4)

print(f"Combined JSON written to {output_file}")

