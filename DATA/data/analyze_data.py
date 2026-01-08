import json
import numpy as np
from collections import Counter
import os
import argparse # Added argparse

# --- Helper Functions ---

def is_numerical(series):
    """Check if a series can be treated as numerical."""
    try:
        # Attempt to convert a sample to float. If it works for one, assume for all.
        if series:
            float(series[0])
        return True
    except (ValueError, TypeError):
        return False

def get_numerical_stats(series):
    """Calculate descriptive statistics for a numerical series."""
    try:
        # Convert all items to float, filtering out potential non-numeric anomalies
        numeric_series = [float(x) for x in series if x is not None]
        if not numeric_series:
            return None
        
        # Using numpy for robust calculations
        arr = np.array(numeric_series)
        stats = {
            "count": len(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "mean": np.mean(arr),
            "std": np.std(arr),
            "25%": np.percentile(arr, 25),
            "50%": np.percentile(arr, 50),
            "75%": np.percentile(arr, 75),
            "90%": np.percentile(arr, 90),
        }
        return stats
    except (ValueError, TypeError):
        # This case should ideally not be hit if is_numerical is used correctly
        return None


def get_categorical_stats(series, top_n=10):
    """Calculate frequency distribution for a categorical series."""
    if not series or not isinstance(series, list):
        return {"count": 0, "unique_values": 0, "frequencies": {}}
        
    count = len(series)
    counter = Counter(series)
    unique_values = len(counter)
    # Get the most common items
    most_common = dict(counter.most_common(top_n))
    
    return {
        "count": count,
        "unique_values": unique_values,
        "frequencies": most_common,
    }

# --- Main Analysis Logic ---

def analyze_datasets(data1, data2, name1="file1", name2="file2"):
    """Main function to run the comparative analysis."""
    
    # Get a set of all protocols from both datasets
    all_protocols = set(data1.keys()) | set(data2.keys())
    
    analysis_results = {}

    for protocol in sorted(list(all_protocols)):
        print(f"--- Analyzing Protocol: {protocol} ---")
        analysis_results[protocol] = {"fields": {}}
        
        protocol_data1 = data1.get(protocol, {})
        protocol_data2 = data2.get(protocol, {})
        
        # Get a set of all fields for this protocol
        all_fields = set(protocol_data1.keys()) | set(protocol_data2.keys())
        
        for field in sorted(list(all_fields)):
            series1 = protocol_data1.get(field, [])
            series2 = protocol_data2.get(field, [])

            # Heuristic to determine field type based on data1 first, then data2
            series_sample = series1 if series1 else series2
            
            field_type = "numerical" if is_numerical(series_sample) else "categorical"
            
            # Special override for fields that look numerical but are categorical
            if field.lower() in ["sip", "dip", "smac", "dmac", "sport", "dport", "sp", "dp"]:
                 field_type = "categorical"
            # Protocol is always categorical
            if field.lower() == "protocol":
                 field_type = "categorical"

            stats1_key = f"{name1}_stats"
            stats2_key = f"{name2}_stats"

            analysis_results[protocol]["fields"][field] = {
                "type": field_type,
                stats1_key: None,
                stats2_key: None,
            }

            if field_type == "numerical":
                stats1 = get_numerical_stats(series1)
                stats2 = get_numerical_stats(series2)
                analysis_results[protocol]["fields"][field][stats1_key] = stats1
                analysis_results[protocol]["fields"][field][stats2_key] = stats2
            else: # Categorical
                stats1 = get_categorical_stats(series1)
                stats2 = get_categorical_stats(series2)
                analysis_results[protocol]["fields"][field][stats1_key] = stats1
                analysis_results[protocol]["fields"][field][stats2_key] = stats2

    return analysis_results


def main():
    """Loads data, runs analysis, and prints or saves results."""
    parser = argparse.ArgumentParser(description="Compare two JSON datasets.")
    parser.add_argument("--file1", required=True, help="Path to the first JSON file.")
    parser.add_argument("--file2", required=True, help="Path to the second JSON file.")
    parser.add_argument("--output", help="Optional path to save the output JSON file.")
    args = parser.parse_args()

    try:
        with open(args.file1, "r", encoding="utf-8") as f:
            data1 = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {args.file1}: {e}")
        return

    try:
        with open(args.file2, "r", encoding="utf-8") as f:
            data2 = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {args.file2}: {e}")
        return

    # Get file names without extension to use as labels
    name1 = os.path.splitext(os.path.basename(args.file1))[0]
    name2 = os.path.splitext(os.path.basename(args.file2))[0]

    # Run the analysis
    results = analyze_datasets(data1, data2, name1, name2)

    # Output the results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Analysis results saved to {args.output}")
    else:
        # Pretty print the results to console
        print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
