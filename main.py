import os
import dspy
import argparse
import pandas as pd
from dotenv import load_dotenv

from .formatting.formatting_report import format_csv
from .create_csv.lung import lung_csv
from .create_csv.large_airway import large_airway_csv
from .create_csv.mediastinum import mediastinum_csv
from .create_csv.heart_and_vessel import heart_and_vessel_csv
from .create_csv.abdomen import abdomen_csv
from .create_csv.osseous_structure import osseous_structure_csv

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False, default=f'{base_path}/result/input/input.csv', help='Input file path')
    parser.add_argument('--format', type=str, required=False, default=f'{base_path}/result/format', help='Format folder path')
    parser.add_argument('--output', type=str, required=False, default=f'{base_path}/result/output', help='Output root path')
    
    args = parser.parse_args()
    
    # OpenAI
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    os.environ['OPENAI_API_KEY'] = api_key
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'], temperature=1.0, max_tokens=5000)

    dspy.configure(lm=lm)
    
    report_df = pd.read_csv(args.input)
    
    # Create Folder
    os.makedirs(args.format, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    # Format
    format_csv(f"{args.format}/format.csv", report_df)
    format_df = pd.read_csv(f"{args.format}/format.csv")
    
    # Create CSV files
    lung_csv(f"{args.output}/lung_result.csv", format_df)
    large_airway_csv(f"{args.output}/large_airway_result.csv", format_df)
    mediastinum_csv(f"{args.output}/mediastinum_result.csv", format_df)
    heart_and_vessel_csv(f"{args.output}/heart_and_vessel_result.csv", format_df)
    abdomen_csv(f"{args.output}/abdomen_result.csv", format_df)
    osseous_structure_csv(f"{args.output}/osseous_structure_result.csv", format_df)
    
    print("CSV files created successfully.")
    cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
    print("Total cost:", cost)

if __name__ == '__main__':
    main()
