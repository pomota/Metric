import pandas as pd
import dspy
from tqdm import tqdm
import os

from .formatting_prompt import *

def format_csv(save_path, report_df):
    columns = ['id', 'original_report', 'lung_report', 'large_airway_report', 'mediastinum_report', 'heart_and_vessel_report', 'abdomen_report', 'osseous_structure_report']
    format_df = pd.DataFrame(columns=columns)

    format_df['id'] = report_df['id']
    format_df['original_report'] = report_df['report']

    # OpenAI
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'], temperature=1.0, max_tokens=5000)

    dspy.configure(lm=lm)

    lung_cot = dspy.ChainOfThought(Extract_LungParenchyma)
    airway_cot = dspy.ChainOfThought(Extract_Airways)
    mediastinum_cot = dspy.ChainOfThought(Extract_Mediastinum)
    heart_cot = dspy.ChainOfThought(Extract_HeartAndGreatVessels)
    abdomen_cot = dspy.ChainOfThought(Extract_Abdomen)
    osseous_cot = dspy.ChainOfThought(Extract_OsseousStructures)
    
    class Formatter(dspy.Module):
        def __init__(self):
            self.lung_cot = lung_cot
            self.airway_cot = airway_cot
            self.mediastinum_cot = mediastinum_cot
            self.heart_cot = heart_cot
            self.abdomen_cot = abdomen_cot
            self.osseous_cot = osseous_cot
        
        def forward(self, report):
            # Extract the relevant sections from the report
            lung_parenchyma = self.lung_cot(report=report).sentences
            airways = self.airway_cot(report=report).sentences
            mediastinum = self.mediastinum_cot(report=report).sentences
            heart_and_great_vessels = self.heart_cot(report=report).sentences
            abdomen = self.abdomen_cot(report=report).sentences
            osseous_structures = self.osseous_cot(report=report).sentences
            
            return {
                'lung_report': lung_parenchyma,
                'large_airway_report': airways,
                'mediastinum_report': mediastinum,
                'heart_and_vessel_report': heart_and_great_vessels,
                'abdomen_report': abdomen,
                'osseous_structure_report': osseous_structures
            }
            
    formatter_module = Formatter()

    print("Formatting report...")
    for idx, row in tqdm(report_df.iterrows()):
        id = row['id']        
        report = row['report']
        
        formatted_report = formatter_module(report)
        
        for key in formatted_report:
            format_df.loc[format_df['id']==id, key] = formatted_report[key]
        
    # OpenAI 비용 계산
    cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
    print("Total cost:", cost)

    # CSV로 저장
    format_df.to_csv(save_path, index=False)

    print(f"File saved to {save_path}")