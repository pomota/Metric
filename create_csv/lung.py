import os
import pandas as pd
from tqdm import tqdm
import dspy
from glob import glob

from ..prompt.lung_prompt import *
        

def lung_csv(save_path, report_df):
    # Disease classifier
    lung_disease_classifier = Lung_Disease_Classifier()

    # Locator
    rl_locator = dspy.ChainOfThought(Locator_RL)
    left_lobe_locator = dspy.ChainOfThought(Locator_Left_Lobes)
    right_lobe_locator = dspy.ChainOfThought(Locator_Right_Lobes)

    # Counter
    counter = dspy.ChainOfThought(Counter)
    
    # Nodule, Mass 제외
    disease_list = ['Nodule', 'Mass', 'Pleural Effusion', 'Consolidation', 'Atelectasis', 'Pneumothorax', 'Ground Glass Opacity', 'Emphysema', 'Mosaic Attenuation', 'Bronchiectasis', 'Interlobular Septal Thickening']

    columns = ['id', 'lung_report']
    for disease_name in disease_list:
        columns.append(f"{disease_name}_presence")
        columns.append(f"{disease_name}_lul")
        columns.append(f"{disease_name}_lll")
        columns.append(f"{disease_name}_left_unspecified")
        columns.append(f"{disease_name}_rul")
        columns.append(f"{disease_name}_rml")
        columns.append(f"{disease_name}_rll")
        columns.append(f"{disease_name}_right_unspecified")
        columns.append(f"{disease_name}_unspecified")
        
        if disease_name in ['Nodule', 'Mass']:
            columns.append(f"{disease_name}_single")
            columns.append(f"{disease_name}_multiple")
    
    result_df = pd.DataFrame(columns=columns)
    result_df['id'] = report_df['id']
    result_df['lung_report'] = report_df['lung_report']
    
    # 모든 값을 0으로 초기화
    for col in columns:
        if col != 'id' and col != 'lung_report':  # id, report 열은 건너뛰기
            result_df[col] = 0
        
    for idx, row in tqdm(report_df.iterrows()):
        id = row['id']
        report = row['lung_report']
        
        disease_classifier_result = lung_disease_classifier(report)
        
        for disease_name in disease_list:
            if int(disease_classifier_result['abnormality_presence'][disease_name]) == 0:
                continue
            
            disease_lesion_sentence = disease_classifier_result['lesion_sentence'][disease_name]
            
            result_df.loc[result_df['id']==id, f"{disease_name}_presence"] = 1
            
            rl_result = rl_locator(report=report, abnormality_class=disease_name)
            
            if int(rl_result.left)==1:
                left_lobe_locator_result = left_lobe_locator(sentence=disease_lesion_sentence, abnormality_class=disease_name)
                
                if int(left_lobe_locator_result.left_upper_lobe)==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_lul"] = 1
                    
                if int(left_lobe_locator_result.left_lower_lobe)==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_lll"] = 1
                
                if int(left_lobe_locator_result.unspecified)==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_left_unspecified"] = 1
                    
            if int(rl_result.right)==1:
                right_lobe_locator_result = right_lobe_locator(sentence=disease_lesion_sentence, abnormality_class=disease_name)
                
                if int(right_lobe_locator_result.right_upper_lobe)==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_rul"] = 1
                    
                if int(right_lobe_locator_result.right_middle_lobe)==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_rml"] = 1
  
                if int(right_lobe_locator_result.right_lower_lobe)==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_rll"] = 1

                if int(right_lobe_locator_result.unspecified)==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_right_unspecified"] = 1
                
            if int(rl_result.unspecified)==1:
                result_df.loc[result_df['id']==id, f"{disease_name}_unspecified"] = 1
            
            if disease_name in ['Nodule', 'Mass']:
                # Count 추가하기
                counter_result = counter(report=report, abnormality_class=disease_name)
                
                if counter_result.single==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_single"] = 1
                
                if counter_result.multiple==1:
                    result_df.loc[result_df['id']==id, f"{disease_name}_multiple"] = 1

    result_df.to_csv(save_path, index=False)
    print("Lung CSV file created successfully.")