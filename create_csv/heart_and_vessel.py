import pandas as pd
import os
import dspy
from tqdm import tqdm

from ..prompt.heart_and_vessel_prompt import *


def heart_and_vessel_csv(save_path, report_df):
    # Disease classifier
    heart_and_vessel_disease_classifier = Heart_and_Vessel_Disease_Classifier()
    
    # Locator
    pe_loctor = dspy.ChainOfThought(Locator_Pulmonary_Embolism)
    
    columns = [
        'id', 'heart_and_vessel_report',
        'Aortic_Aneurysm_presence', 'Aortic_Dilatation_presence', 'Aortic_Dissection_presence', 'Pulmonary_Artery_Enlargement_presence',
        'Pulmonary_Embolism_presence', 'Pulmonary_Embolism_right', 'Pulmonary_Embolism_left', 'Pulmonary_Embolism_main', 'Pulmonary_Embolism_unspecified',
        'Cardiomegaly_presence', 'Pericardial_Effusion_presence', 'Cardiac_Mass_presence', 'Coronary_Artery_Wall_Calcification_presence', 'Arterial_Calcification_presence'
    ] #dilation -> dilatation
    
    result_df = pd.DataFrame(columns=columns)
    result_df['id'] = report_df['id']
    result_df['heart_and_vessel_report'] = report_df['heart_and_vessel_report']
    
    # 모든 값을 0으로 초기화
    for col in columns:
        if col != 'id' and col != 'heart_and_vessel_report':  # id, report 열은 건너뛰기
            result_df[col] = 0
            
    # Csv 생성
    for idx, row in tqdm(report_df.iterrows()):
        id = row['id']
        report = row['heart_and_vessel_report']
        
        disease_classifier_result = heart_and_vessel_disease_classifier(report=report)
        
        # Aortic Aneurysm
        if int(disease_classifier_result['Aortic_Aneurysm'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Aortic_Aneurysm_presence"] = 1
        
        # Aortic Dilatation
        if int(disease_classifier_result['Aortic_Dilatation'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Aortic_Dilatation_presence"] = 1
        
        # Aortic Dissection
        if int(disease_classifier_result['Aortic_Dissection'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Aortic_Dissection_presence"] = 1
        
        # Pulmonary Artery Enlargement
        if int(disease_classifier_result['Pulmonary_Artery_Enlargement'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Pulmonary_Artery_Enlargement_presence"] = 1
        
        # Pulmonary Embolism
        if int(disease_classifier_result['Pulmonary_Embolism'].abnormality_presence) == 1:
            pulmonary_embolism_sentence = disease_classifier_result['Pulmonary_Embolism'].lesion_sentence
            pe_locator_result = pe_loctor(sentence=pulmonary_embolism_sentence, abnormality_class="Pulmonary_Embolism") #report -> sentence
            
            # Presence
            result_df.loc[result_df['id']==id, "Pulmonary_Embolism_presence"] = 1
            
            # Locator
            if int(pe_locator_result.right) == 1:
                result_df.loc[result_df['id']==id, 'Pulmonary_Embolism_right'] = 1
            if int(pe_locator_result.left) == 1:
                result_df.loc[result_df['id']==id, 'Pulmonary_Embolism_left'] = 1
            if int(pe_locator_result.main) == 1:
                result_df.loc[result_df['id']==id, 'Pulmonary_Embolism_main'] = 1
            if int(pe_locator_result.unspecified) == 1:
                result_df.loc[result_df['id']==id, 'Pulmonary_Embolism_unspecified'] = 1
            
        # Cardiomegaly
        if int(disease_classifier_result['Cardiomegaly'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Cardiomegaly_presence"] = 1
        
        # Pericardial Effusion
        if int(disease_classifier_result['Pericardial_Effusion'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Pericardial_Effusion_presence"] = 1
        
        # Cardiac Mass
        if int(disease_classifier_result['Cardiac_Mass'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Cardiac_Mass_presence"] = 1
        
        # Coronary Artery Wall Calcification
        if int(disease_classifier_result['Coronary_Artery_Wall_Calcification'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Coronary_Artery_Wall_Calcification_presence"] = 1
        
        # Arterial Calcification
        if int(disease_classifier_result['Arterial_Calcification'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, "Arterial_Calcification_presence"] = 1
    
    # Save to CSV
    result_df.to_csv(save_path, index=False)
    print("Heart and vessel CSV file created successfully.")