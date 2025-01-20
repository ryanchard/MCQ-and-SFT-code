import json
from pathlib import Path

# Directory containing the JSON files
input_directory = "FrankieFiles"
output_file = "combined.json"

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

# Write the combined list to the output file
with open(output_file, "w") as output:
    json.dump(combined_json_list, output, indent=4)

print(f"Combined JSON written to {output_file}")

