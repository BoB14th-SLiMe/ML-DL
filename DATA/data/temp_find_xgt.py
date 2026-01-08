
import json
from collections import Counter

def analyze_xgt_response_values(jsonl_path):
    """
    Analyzes the values within 'xgt_fen.data' for the top 5 most frequent
    'xgt_fen.vars' addresses in response packets.
    """
    # The top 5 addresses identified in the previous analysis
    top_5_vars = [
        "%DB001000",
        "%DB001196",
        "%MB000002",
        "%DB001194",
        "%DB001046"
    ]

    # A dictionary to hold a Counter for each address
    results = {var: Counter() for var in top_5_vars}
    
    line_count = 0
    response_packet_count = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Process only xgt_fen response packets
                if obj.get("protocol") == "xgt_fen" and obj.get("dir") == "response":
                    response_packet_count += 1
                    address = obj.get("xgt_fen.vars")
                    
                    # Check if the address is one we are tracking
                    if address in top_5_vars:
                        data_value = obj.get("xgt_fen.data")
                        if data_value is not None:
                            results[address][data_value] += 1
        
        print(f"Processed {line_count} lines and found {response_packet_count} 'xgt_fen' response packets.")
        print("\\n--- Analysis of 'xgt_fen.data' values for each address ---")
        
        for address, counter in results.items():
            print(f"\\n[Address: {address}]")
            if not counter:
                print("  No corresponding data values found in responses.")
                continue
            
            print("  Found the following unique data values and their frequencies:")
            # Sort by count descending
            for value, count in counter.most_common():
                print(f"    - Value: {value}, Count: {count}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {jsonl_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    path_to_check = "C:\\Users\\USER\\Desktop\\bob 프로젝트\\AI\\ML-DL\\DATA\\data\\1128 dataset\\normal_merged.jsonl"
    analyze_xgt_response_values(path_to_check)
