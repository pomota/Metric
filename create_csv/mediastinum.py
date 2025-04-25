import pandas as pd
import dspy
import os
from tqdm import tqdm

from ..prompt.mediastinum_prompt import *

def mediastinum_csv(save_path, report_df):
    # Disease classifier
    mediastinum_disease_classifier = Mediastinum_Disease_Classifier()
    
    # Locator
    locator_mediastinal_mass = dspy.ChainOfThought(Locator_Mediastinal_Mass)
    locator_lymphadenopathy = dspy.ChainOfThought(Locator_Lymphadenopathy)
    
    # Counter
    counter_esophageal_mass = dspy.ChainOfThought(Counter_Esophageal_Mass)

    columns = [
        'id', 'mediastinum_report',
        'Mediastinal_Mass_presence', 'Mediastinal_Mass_anterior', 'Mediastinal_Mass_middle', 'Mediastinal_Mass_posterior', 'Mediastinal_Mass_unspecified',
        
        'Lymphadenopathy_presence', 'Lymphadenopathy_supraclavicular', 'Lymphadenopathy_upper_paratracheal', 'Lymphadenopathy_prevascular', 'Lymphadenopathy_prevertebral',
        'Lymphadenopathy_lower_paratracheal', 'Lymphadenopathy_subaortic', 'Lymphadenopathy_paraaortic', 'Lymphadenopathy_subcarinal', 'Lymphadenopathy_paraesophageal',
        'Lymphadenopathy_hilar', 'Lymphadenopathy_unspecified',
        
        'Esophageal_Mass_presence', 'Esophageal_Mass_single', 'Esophageal_Mass_multiple',
        
        'Pneumomediastinum_presence'
    ]
    
    result_df = pd.DataFrame(columns=columns)
    result_df['id'] = report_df['id']
    result_df['mediastinum_report'] = report_df['mediastinum_report']
    
    # 모든 값을 0으로 초기화
    for col in columns:
        if col != 'id' and col != 'mediastinum_report':  # id, report 열은 건너뛰기
            result_df[col] = 0
    
    # Csv 생성
    for idx, row in tqdm(report_df.iterrows()):
        id = row['id']
        report = row['mediastinum_report']

        disease_classifier_result = mediastinum_disease_classifier(report=report)
        
        # Mediastinal mass
        if int(disease_classifier_result['Mediastinal_Mass'].abnormality_presence) == 1:
            mediastinal_mass_sentence = disease_classifier_result['Mediastinal_Mass'].lesion_sentence
            locator_mediastinal_result = locator_mediastinal_mass(lesion_sentence=mediastinal_mass_sentence)
            
            # Presence
            result_df.loc[result_df['id']==id, "Mediastinal_Mass_presence"] = 1

            # Mediastinal mass locator
            if int(locator_mediastinal_result.anterior) == 1:
                result_df.loc[result_df['id']==id, 'Mediastinal_Mass_anterior'] = 1
            if int(locator_mediastinal_result.middle) == 1:
                result_df.loc[result_df['id']==id, 'Mediastinal_Mass_middle'] = 1
            if int(locator_mediastinal_result.posterior) == 1:
                result_df.loc[result_df['id']==id, 'Mediastinal_Mass_posterior'] = 1
            if int(locator_mediastinal_result.unspecified) == 1:
                result_df.loc[result_df['id']==id, 'Mediastinal_Mass_unspecified'] = 1

        # Lymphadenopathy
        if int(disease_classifier_result['Lymphadenopathy'].abnormality_presence) == 1:
            lymphadenopathy_sentence = disease_classifier_result['Lymphadenopathy'].lesion_sentence
            locator_lymphadenopathy_result = locator_lymphadenopathy(lesion_sentence=lymphadenopathy_sentence)

            result_df.loc[result_df['id']==id, "Lymphadenopathy_presence"] = 1

            # Lymphadenopathy locator
            if int(locator_lymphadenopathy_result.supraclavicular) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_supraclavicular'] = 1
            if int(locator_lymphadenopathy_result.upper_paratracheal) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_upper_paratracheal'] = 1
            if int(locator_lymphadenopathy_result.prevascular) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_prevascular'] = 1
            if int(locator_lymphadenopathy_result.prevertebral) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_prevertebral'] = 1
            if int(locator_lymphadenopathy_result.lower_paratracheal) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_lower_paratracheal'] = 1
            if int(locator_lymphadenopathy_result.subaortic) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_subaortic'] = 1
            if int(locator_lymphadenopathy_result.paraaortic) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_paraaortic'] = 1
            if int(locator_lymphadenopathy_result.subcarinal) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_subcarinal'] = 1
            if int(locator_lymphadenopathy_result.paraesophageal) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_paraesophageal'] = 1
            if int(locator_lymphadenopathy_result.hilar) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_hilar'] = 1
            if int(locator_lymphadenopathy_result.unspecified) == 1:
                result_df.loc[result_df['id']==id, 'Lymphadenopathy_unspecified'] = 1

        # Esophageal Mass
        if int(disease_classifier_result['Esophageal_Mass'].abnormality_presence) == 1:
            esophageal_mass_sentence = disease_classifier_result['Esophageal_Mass'].lesion_sentence
            counter_esophageal_mass_result = counter_esophageal_mass(lesion_sentence=esophageal_mass_sentence)

            result_df.loc[result_df['id']==id, "Esophageal_Mass_presence"] = 1    

            if int(counter_esophageal_mass_result.single) == 1:
                result_df.loc[result_df['id']==id, 'Esophageal_Mass_single'] = 1
            if int(counter_esophageal_mass_result.multiple) == 1:
                result_df.loc[result_df['id']==id, 'Esophageal_Mass_multiple'] = 1            
                
        # Pneumomediastinum
        if int(disease_classifier_result['Pneumomediastinum'].abnormality_presence) == 1:
            result_df.loc[result_df['id']==id, 'Pneumomediastinum_presence'] = 1

    # Sort df
    result_df = result_df.sort_values(by='id')
    
    result_df.to_csv(save_path, index=False)
    print("Mediastinum CSV file created successfully.")