import pandas as pd
import dspy
import os
import random
from tqdm import tqdm

from ..prompt.large_airway_prompt import *

def large_airway_csv(save_path, report_df):
    # Disease classifier
    large_airway_disease_classifier = Large_Airway_Disease_Classifier()
    
    # Locator
    locator_endobronchial_mass = dspy.ChainOfThought(Locator_Endobronchial_Mass)
    
    columns = [
        'id', 'large_airway_report',
        'Tracheal_Stenosis_presence',
        'Endotracheal_Mass_presence', 'Endotracheal_Mass_single', 'Endotracheal_Mass_multiple',
        'Endobronchial_Mass_presence', 'Endobronchial_Mass_left', 'Endobronchial_Mass_right', 'Endobronchial_Mass_unspecified',
        'Endobronchial_Mass_single', 'Endobronchial_Mass_multiple'
    ]
    
    result_df = pd.DataFrame(columns=columns)
    result_df['id'] = report_df['id']
    result_df['large_airway_report'] = report_df['large_airway_report']
    
    # 모든 값을 0으로 초기화
    for col in columns:
        if col != 'id' and col != 'large_airway_report':  # id, report 열은 건너뛰기
            result_df[col] = 0
    
    for idx, row in tqdm(report_df.iterrows()):
        id = row['id']
        report = row['large_airway_report']
        
        disease_classifier_result = large_airway_disease_classifier(report=report)

        # Tracheal Stenosis
        if int(disease_classifier_result['tracheal_stenosis'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Tracheal_Stenosis_presence"] = 1
        
        # Endotracheal Mass
        if int(disease_classifier_result['endotracheal_mass'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Endotracheal_Mass_presence"] = 1
            
            if int(disease_classifier_result['endotracheal_mass'].mass_count_single) == 1:
                result_df.loc[result_df['id']==id, "Endotracheal_Mass_single"] = 1
            if int(disease_classifier_result['endotracheal_mass'].mass_count_multiple) == 1:
                result_df.loc[result_df['id']==id, "Endotracheal_Mass_multiple"] = 1
        
        # Endobronchial Mass
        if int(disease_classifier_result['endobronchial_mass'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Endobronchial_Mass_presence"] = 1
            endobronchial_mass_sentence = disease_classifier_result['endobronchial_mass'].lesion_sentence
            locator_endobronchial_result = locator_endobronchial_mass(lesion_sentence=endobronchial_mass_sentence, abnormality_class="Endobronchial_Mass")
            
            if int(locator_endobronchial_result.left_main) == 1:
                result_df.loc[result_df['id']==id, 'Endobronchial_Mass_left'] = 1
            if int(locator_endobronchial_result.right_main) == 1:
                result_df.loc[result_df['id']==id, 'Endobronchial_Mass_right'] = 1
            if int(locator_endobronchial_result.unspecified) == 1:
                result_df.loc[result_df['id']==id, 'Endobronchial_Mass_unspecified'] = 1

            if int(disease_classifier_result['endobronchial_mass'].mass_count_single) == 1:
                result_df.loc[result_df['id']==id, "Endobronchial_Mass_single"] = 1
            if int(disease_classifier_result['endobronchial_mass'].mass_count_multiple) == 1:
                result_df.loc[result_df['id']==id, "Endobronchial_Mass_multiple"] = 1

    # Save to CSV
    result_df.to_csv(save_path, index=False)
    print("Large Airway CSV file created successfully.")