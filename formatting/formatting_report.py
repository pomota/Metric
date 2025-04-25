import pandas as pd
import dspy
from tqdm import tqdm
import os
import concurrent.futures
from functools import partial

from .formatting_prompt import *

def process_report(report_id, report_text, lung_cot, airway_cot, mediastinum_cot, heart_cot, abdomen_cot, osseous_cot):
    """Process a single report with all classifiers"""
    try:
        # Extract the relevant sections from the report
        lung_parenchyma = lung_cot(report=report_text).sentences
        airways = airway_cot(report=report_text).sentences
        mediastinum = mediastinum_cot(report=report_text).sentences
        heart_and_great_vessels = heart_cot(report=report_text).sentences
        abdomen = abdomen_cot(report=report_text).sentences
        osseous_structures = osseous_cot(report=report_text).sentences
        
        return {
            'id': report_id,
            'original_report': report_text,
            'lung_report': lung_parenchyma,
            'large_airway_report': airways,
            'mediastinum_report': mediastinum,
            'heart_and_vessel_report': heart_and_great_vessels,
            'abdomen_report': abdomen,
            'osseous_structure_report': osseous_structures
        }
    except Exception as e:
        print(f"Error processing report {report_id}: {e}")
        # Return a row with the ID and original report, but empty formatted sections
        return {
            'id': report_id,
            'original_report': report_text,
            'lung_report': "",
            'large_airway_report': "",
            'mediastinum_report': "",
            'heart_and_vessel_report': "",
            'abdomen_report': "",
            'osseous_structure_report': ""
        }

def format_csv(save_path, report_df, max_workers=None):
    """
    Process reports in parallel using ThreadPoolExecutor
    
    Args:
        save_path: Path to save the formatted CSV
        report_df: DataFrame containing reports to process
        max_workers: Number of parallel workers (default: None, which uses CPU count)
    """
    columns = ['id', 'original_report', 'lung_report', 'large_airway_report', 
               'mediastinum_report', 'heart_and_vessel_report', 'abdomen_report', 
               'osseous_structure_report']

    # OpenAI
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'], temperature=1.0, max_tokens=5000)
    dspy.configure(lm=lm)

    # Initialize classifiers
    lung_cot = dspy.ChainOfThought(Extract_LungParenchyma)
    airway_cot = dspy.ChainOfThought(Extract_Airways)
    mediastinum_cot = dspy.ChainOfThought(Extract_Mediastinum)
    heart_cot = dspy.ChainOfThought(Extract_HeartAndGreatVessels)
    abdomen_cot = dspy.ChainOfThought(Extract_Abdomen)
    osseous_cot = dspy.ChainOfThought(Extract_OsseousStructures)
    
    print("Formatting reports in parallel...")
    
    # Create a partial function with the classifiers already bound
    process_func = partial(
        process_report,
        lung_cot=lung_cot,
        airway_cot=airway_cot,
        mediastinum_cot=mediastinum_cot,
        heart_cot=heart_cot,
        abdomen_cot=abdomen_cot,
        osseous_cot=osseous_cot
    )
    
    results = []
    
    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(process_func, row['id'], row['report']): row['id'] 
            for _, row in report_df.iterrows()
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(future_to_id), desc="Formatting reports"):
            report_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Report {report_id} generated an exception: {e}")
    
    # Convert results to DataFrame and save
    format_df = pd.DataFrame(results, columns=columns)
    
    # OpenAI cost calculation
    cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
    print("Total cost:", cost)

    # Sort df
    format_df = format_df.sort_values(by='id')
    
    # Save to CSV
    format_df.to_csv(save_path, index=False)
    print(f"File saved to {save_path}")
    
    return format_df