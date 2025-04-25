import pandas as pd
import dspy
import os
from tqdm import tqdm

from ..prompt.abdomen_prompt import *

def abdomen_csv(save_path, report_df):
    # Disease classifier
    abdomen_disease_classifier = Abdomen_Disease_Classifier()

    # Locator
    kidney_locator = dspy.ChainOfThought(Locator_Kidney_RL)
    adrenal_locator = dspy.ChainOfThought(Locator_adrenal_RL)

    # Counter
    kidney_count = dspy.ChainOfThought(Counter_Kidneycyst)
    Liver_count = dspy.ChainOfThought(Counter_Livercyst)

    columns = [
        'id', 'abdomen_report',
        
        'Kidney_Cyst_presence', 'Kidney_Cyst_right', 'Kidney_Cyst_left', 'Kidney_Cyst_unspecified',
        'Kidney_Cyst_single', 'Kidney_Cyst_multiple',
        
        'Adrenal_Mass_presence', 'Adrenal_Mass_right', 'Adrenal_Mass_left', 'Adrenal_Mass_unspecified',
        
        'Liver_Cyst_presence', 'Liver_Cyst_single', 'Liver_Cyst_multiple',
        
        'Gallstone_presence', 'Hiatal_Hernia_presence', 'Pneumoperitoneum_presence'
    ]

    result_df = pd.DataFrame(columns=columns)
    result_df['id'] = report_df['id']
    result_df['abdomen_report'] = report_df['abdomen_report']

    # 모든 값을 0으로 초기화
    for col in columns:
        if col != 'id' and col != 'abdomen_report':  # id, report 열은 건너뛰기
            result_df[col] = 0
    
    # Csv 생성      
    for idx, row in tqdm(report_df.iterrows()):
        id = row['id']
        report = row['abdomen_report']

        disease_classifier_result = abdomen_disease_classifier(report=report)

        # Kidney Cyst
        if int(disease_classifier_result['kidney_cyst'].abnormality_presence) == 1:
            kidney_cyst_sentence = disease_classifier_result['kidney_cyst'].lesion_sentence
            locator_kidney_cyst_result = kidney_locator(report=kidney_cyst_sentence, abnormality_class="kidney_cyst")
            counter_kidney_cyst_result = kidney_count(lesion_sentence=kidney_cyst_sentence)
            
            # Presence
            result_df.loc[result_df['id']==id, "Kidney_Cyst_presence"] = 1
            
            # Locator
            if int(locator_kidney_cyst_result.right) == 1:
                result_df.loc[result_df['id']==id, 'Kidney_Cyst_right'] = 1
            if int(locator_kidney_cyst_result.left) == 1:
                result_df.loc[result_df['id']==id, 'Kidney_Cyst_left'] = 1
            if int(locator_kidney_cyst_result.unspecified) == 1:
                result_df.loc[result_df['id']==id, 'Kidney_Cyst_unspecified'] = 1
            
            if int(counter_kidney_cyst_result.single) == 1:
                result_df.loc[result_df['id']==id, 'Kidney_Cyst_single'] = 1
            if int(counter_kidney_cyst_result.multiple) == 1:
                result_df.loc[result_df['id']==id, 'Kidney_Cyst_multiple'] = 1
            
        # Adrenal Mass
        if int(disease_classifier_result['adrenal_mass'].abnormality_presence) == 1:
            adrenal_mass_sentence = disease_classifier_result['adrenal_mass'].lesion_sentence
            locator_adrenal_result = adrenal_locator(report=adrenal_mass_sentence, abnormality_class="adrenal_mass")
            
            # Presence
            result_df.loc[result_df['id']==id, "Adrenal_Mass_presence"] = 1
            
            # Locator
            if int(locator_adrenal_result.right) == 1:
                result_df.loc[result_df['id']==id, 'Adrenal_Mass_right'] = 1
            if int(locator_adrenal_result.left) == 1:
                result_df.loc[result_df['id']==id, 'Adrenal_Mass_left'] = 1
            if int(locator_adrenal_result.unspecified) == 1:
                result_df.loc[result_df['id']==id, 'Adrenal_Mass_unspecified'] = 1
        
        # Liver Cyst
        if int(disease_classifier_result['liver_cyst'].abnormality_presence) == 1:
            liver_cyst_sentence = disease_classifier_result['liver_cyst'].lesion_sentence
            counter_liver_cyst_result = Liver_count(lesion_sentence=liver_cyst_sentence)
            
            # Presence
            result_df.loc[result_df['id']==id, "Liver_Cyst_presence"] = 1
            
            if int(counter_liver_cyst_result.single) == 1:
                result_df.loc[result_df['id']==id, 'Liver_Cyst_single'] = 1
            if int(counter_liver_cyst_result.multiple) == 1:
                result_df.loc[result_df['id']==id, 'Liver_Cyst_multiple'] = 1
        
        # Gall stone
        if int(disease_classifier_result['gallstone'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Gallstone_presence"] = 1
        # Hiatal Hernia
        if int(disease_classifier_result['hiatal_hernia'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Hiatal_Hernia_presence"] = 1
        # Pneumoperitoneum
        if int(disease_classifier_result['pneumoperitoneum'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Pneumoperitoneum_presence"] = 1
    
    # Sort df
    result_df = result_df.sort_values(by='id')
    
    # Save to CSV
    result_df.to_csv(save_path, index=False)
    print("Abdomen CSV file created successfully.")