import os
import dspy
import argparse
import pandas as pd
from dotenv import load_dotenv
import asyncio
import time
import concurrent.futures
from tqdm import tqdm

# Import processing modules
from .formatting.formatting_report import format_csv  # Assuming you've updated this with the parallel version
from .create_csv.lung import lung_csv
from .create_csv.large_airway import large_airway_csv
from .create_csv.mediastinum import mediastinum_csv
from .create_csv.heart_and_vessel import heart_and_vessel_csv
from .create_csv.abdomen import abdomen_csv
from .create_csv.osseous_structure import osseous_structure_csv

# Import evaluation
from .f1_calculator import calculate_organ_f1


# Create async wrapper for each processing function
async def async_process(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

async def main_async():
    start_time_total = time.time()
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    
    args, _ = parser.parse_known_args()  # Pre-parse to get experiment name
    
    parser.add_argument('--input', type=str, required=False, default=f'{base_path}/result/input/{args.exp}.csv', help='Input file path')
    parser.add_argument('--format', type=str, required=False, default=f'{base_path}/result/format/{args.exp}', help='Format folder path')
    parser.add_argument('--output', type=str, required=False, default=f'{base_path}/result/output/{args.exp}', help='Output root path')
    parser.add_argument('--gt', type=str, required=False, default=f'{base_path}/result/ground_truth', help='Ground truth directory path')
    
    parser.add_argument('--format_workers', type=int, required=False, default=None, help='Number of workers for formatting')
    parser.add_argument('--csv_workers', type=int, required=False, default=6, help='Number of workers for CSV creation')
    
    parser.add_argument('--no-eval', action='store_false', dest='eval', default=True, help='Skip evaluation when specified')
    
    args = parser.parse_args()
    
    # OpenAI setup
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    os.environ['OPENAI_API_KEY'] = api_key
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'], temperature=1.0, max_tokens=5000)

    dspy.configure(lm=lm)
    
    # Create directories
    os.makedirs(args.format, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    # Read input data
    report_df = pd.read_csv(args.input)
    
    # Format reports with parallel processing
    print("Starting report formatting...")
    start_time = time.time()
    
    format_df = format_csv(f"{args.format}/format.csv", report_df, max_workers=args.format_workers)
    
    end_time = time.time()
    print(f"Formatting completed in {end_time - start_time:.2f} seconds.")
    
    # Create CSV files in parallel
    print("Starting CSV creation...")
    start_time = time.time()
    
    # Define tasks to run in parallel using async
    tasks = [
        async_process(lung_csv, f"{args.output}/lung.csv", format_df),
        async_process(large_airway_csv, f"{args.output}/large_airway.csv", format_df),
        async_process(mediastinum_csv, f"{args.output}/mediastinum.csv", format_df),
        async_process(heart_and_vessel_csv, f"{args.output}/heart_and_vessel.csv", format_df),
        async_process(abdomen_csv, f"{args.output}/abdomen.csv", format_df),
        async_process(osseous_structure_csv, f"{args.output}/osseous_structure.csv", format_df)
    ]
    
    # Create a progress bar to track completion
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating CSV files"):
        await task
    
    end_time = time.time()
    print(f"CSV creation completed in {end_time - start_time:.2f} seconds.")
    
    # 평가 수행 (--no-eval 옵션이 없는 경우)
    if args.eval:
        print("\nStarting evaluation... (Add [--no-eval] to skip)")
        start_time = time.time()
        
        # 평가 결과 저장 경로
        metrics_path = f"{args.output}/metrics.json"
        
        # F1 계산 실행
        calculate_organ_f1(args.output, args.gt, metrics_path)
        
        end_time = time.time()
        print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    else:
        print("\nEvaluation skipped.")
    
    # Total time and cost calculation
    end_time_total = time.time()
    print(f"Total processing time: {end_time_total - start_time_total:.2f} seconds")
    
    cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
    print("Total OpenAI cost:", cost)

def main():
    # Run the async main function
    asyncio.run(main_async())

if __name__ == '__main__':
    main()