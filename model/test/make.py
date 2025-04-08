import os
import pandas as pd
import argparse
import random

def create_test_csv(input_filename, output_filename):
    # Use the directory of this script for the file paths
    base_dir = os.path.dirname(__file__)
    input_file_path = os.path.join(base_dir, input_filename)
    
    # Read the input CSV file
    df = pd.read_csv(input_file_path)
    
    # Define the columns to extract
    required_columns = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean']
    
    # Check that all required columns exist in the input file
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print("Error: The following required columns are missing from the input CSV:", missing_cols)
        return
    
    # Determine a random sample size between 10 and 15
    sample_size = random.randint(10, 15)
    print(f"Sampling {sample_size} rows from the input file.")
    
    # If the DataFrame has fewer rows than the sample size, use all rows
    if len(df) < sample_size:
        sample_size = len(df)
    
    # Randomly sample 'sample_size' rows
    df_sample = df.sample(n=sample_size, random_state=random.randint(0, 10000))
    
    # Extract only the required columns
    df_test = df_sample[required_columns]
    
    # Build output file path in the same directory
    output_file_path = os.path.join(base_dir, output_filename)
    
    # Write the new DataFrame to a CSV file without the index column
    df_test.to_csv(output_file_path, index=False)
    print(f"Test CSV file saved to '{output_file_path}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Extract 10 to 15 random samples from TTSWING.csv and save only the specified columns."
    )
    parser.add_argument("--input", default="TTSWING.csv", help="Name of the input CSV file (default: TTSWING.csv)")
    parser.add_argument("--output", default="test.csv", help="Name of the output test CSV file (default: test.csv)")
    
    args = parser.parse_args()
    create_test_csv(args.input, args.output)

if __name__ == "__main__":
    main()
