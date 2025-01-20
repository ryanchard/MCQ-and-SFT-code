#!/usr/bin/env python3

import sys
import json

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_json_file> <output_json_file>")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_json = sys.argv[2]
    
    # Read the JSON data from the input file
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    # Extract only the question and answer keys
    results = []
    for record in data:
        # Assuming each record has keys: "question", "answer", and "text"
        new_record = {
            "question": record.get("question"),
            "answer": record.get("answer")
        }
        results.append(new_record)
    
    # Write the extracted data to the output JSON file
    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
