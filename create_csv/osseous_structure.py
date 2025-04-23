import pandas as pd
import dspy
import os
from tqdm import tqdm

from ..prompt.osseous_structure_prompt import *

def osseous_structure_csv(save_path, report_df):
    # Disease classifier
    osseous_structure_disease_classifier = Osseous_Structure_Disease_Classifier()
    
    # Locator
    locator_rf = dspy.ChainOfThought(Locator_Rib_Fracture)
    locator_vf = dspy.ChainOfThought(Locator_Vertebrae_Fracture) # 수정정
    
    # Onset
    onset_rf = dspy.ChainOfThought(Onset_Rib_Fracture)
    
    rib_numbers = range(1, 13)
    sides = ['right', 'left']
    onset_status = ['new', 'old_healed', 'unspecified']

    columns = ['id', 'osseous_structure_report']
    columns.append('rib_fracture_presence')
    for i in rib_numbers:
        columns.append(f'rib_fracture_right_{i}_presence')
        columns.append(f'rib_fracture_right_{i}_new')
        columns.append(f'rib_fracture_right_{i}_old_healed')
        columns.append(f'rib_fracture_right_{i}_unspecified')
    
    for i in rib_numbers:
        columns.append(f'rib_fracture_left_{i}_presence')
        columns.append(f'rib_fracture_left_{i}_new')
        columns.append(f'rib_fracture_left_{i}_old_healed')
        columns.append(f'rib_fracture_left_{i}_unspecified')
    columns.append('rib_fracture_unspecified')
    
    columns.extend([
        'vertebrae_fracture_presence',
        'vertebrae_fracture_C7',
        'vertebrae_fracture_T1',
        'vertebrae_fracture_T2',
        'vertebrae_fracture_T3',
        'vertebrae_fracture_T4',
        'vertebrae_fracture_T5',
        'vertebrae_fracture_T6',
        'vertebrae_fracture_T7',
        'vertebrae_fracture_T8',
        'vertebrae_fracture_T9',
        'vertebrae_fracture_T10',
        'vertebrae_fracture_T11',
        'vertebrae_fracture_T12',
        'vertebrae_fracture_L1',
        'vertebrae_fracture_L2',
        'vertebrae_fracture_L3',
        'vertebrae_fracture_unspecified'
    ])
    
    result_df = pd.DataFrame(columns=columns)
    result_df['id'] = report_df['id']
    result_df['osseous_structure_report'] = report_df['osseous_structure_report']
    
    # 모든 값을 0으로 초기화
    for col in columns:
        if col != 'id' and col != 'osseous_structure_report':  # id, report 열은 건너뛰기
            result_df[col] = 0
    
    for idx, row in tqdm(report_df.iterrows()):
        id = row['id']
        report = row['osseous_structure_report']
        
        disease_classifier_result = osseous_structure_disease_classifier(report=report)
        
        # Rib fracture                
        if int(disease_classifier_result['abnormality_presence']['rib_fracture']) == 1:
            rf_sentence = disease_classifier_result['lesion_sentence']['rib_fracture']
            locator_rf_result = locator_rf(lesion_sentence=rf_sentence)
            
            result_df.loc[result_df['id']==id, "rib_fracture_presence"] = 1
            
            # Locator + Onset
            if int(locator_rf_result.right1) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_1_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 1st rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_1_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_1_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_1_unspecified'] = 1
            if int(locator_rf_result.right2) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_2_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 2nd rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_2_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_2_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_2_unspecified'] = 1
            if int(locator_rf_result.right3) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_3_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 3rd rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_3_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_3_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_3_unspecified'] = 1
            if int(locator_rf_result.right4) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_4_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 4th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_4_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_4_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_4_unspecified'] = 1
            if int(locator_rf_result.right5) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_5_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 5th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_5_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_5_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_5_unspecified'] = 1
            if int(locator_rf_result.right6) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_6_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 6th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_6_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_6_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_6_unspecified'] = 1
            if int(locator_rf_result.right7) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_7_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 7th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_7_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_7_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_7_unspecified'] = 1
            if int(locator_rf_result.right8) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_8_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 8th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_8_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_8_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_8_unspecified'] = 1
            if int(locator_rf_result.right9) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_9_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 9th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_9_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_9_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_9_unspecified'] = 1
            if int(locator_rf_result.right10) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_10_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 10th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_10_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_10_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_10_unspecified'] = 1
            if int(locator_rf_result.right11) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_11_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 11th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_11_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_11_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_11_unspecified'] = 1
            if int(locator_rf_result.right12) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_right_12_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='right 12th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_12_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_12_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_right_12_unspecified'] = 1

            if int(locator_rf_result.left1) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_1_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 1st rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_1_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_1_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_1_unspecified'] = 1
            if int(locator_rf_result.left2) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_2_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 2nd rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_2_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_2_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_2_unspecified'] = 1
            if int(locator_rf_result.left3) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_3_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 3rd rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_3_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_3_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_3_unspecified'] = 1
            if int(locator_rf_result.left4) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_4_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 4th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_4_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_4_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_4_unspecified'] = 1
            if int(locator_rf_result.left5) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_5_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 5th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_5_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_5_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_5_unspecified'] = 1
            if int(locator_rf_result.left6) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_6_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 6th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_6_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_6_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_6_unspecified'] = 1
            if int(locator_rf_result.left7) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_7_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 7th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_7_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_7_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_7_unspecified'] = 1
            if int(locator_rf_result.left8) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_8_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 8th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_8_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_8_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_8_unspecified'] = 1
            if int(locator_rf_result.left9) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_9_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 9th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_9_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_9_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_9_unspecified'] = 1
            if int(locator_rf_result.left10) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_10_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 10th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_10_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_10_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_10_unspecified'] = 1
            if int(locator_rf_result.left11) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_11_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 11th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_11_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_11_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_11_unspecified'] = 1
            if int(locator_rf_result.left12) == 1:
                result_df.loc[result_df['id']==id, 'rib_fracture_left_12_presence'] = 1
                onset = onset_rf(lesion_sentence=rf_sentence, location='left 12th rib')
                if int(onset.new) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_12_new'] = 1
                if int(onset.old_healed) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_12_old_healed'] = 1
                if int(onset.unspecified) == 1:
                    result_df.loc[result_df['id']==id, f'rib_fracture_left_12_unspecified'] = 1

        
        # Vertebrae fracture
        if int(disease_classifier_result['abnormality_presence']['vertebrae_fracture']) == 1:
            vf_sentence = disease_classifier_result['lesion_sentence']['vertebrae_fracture']
            locator_vf_result = locator_vf(lesion_sentence=vf_sentence)
            result_df.loc[result_df['id']==id, "vertebrae_fracture_presence"] = 1

            if int(locator_vf_result.C7) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_C7'] = 1
            if int(locator_vf_result.T1) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T1'] = 1
            if int(locator_vf_result.T2) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T2'] = 1
            if int(locator_vf_result.T3) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T3'] = 1
            if int(locator_vf_result.T4) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T4'] = 1
            if int(locator_vf_result.T5) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T5'] = 1
            if int(locator_vf_result.T6) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T6'] = 1
            if int(locator_vf_result.T7) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T7'] = 1
            if int(locator_vf_result.T8) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T8'] = 1
            if int(locator_vf_result.T9) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T9'] = 1
            if int(locator_vf_result.T10) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T10'] = 1
            if int(locator_vf_result.T11) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T11'] = 1
            if int(locator_vf_result.T12) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_T12'] = 1
            if int(locator_vf_result.L1) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_L1'] = 1
            if int(locator_vf_result.L2) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_L2'] = 1
            if int(locator_vf_result.L3) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_L3'] = 1
            if int(locator_vf_result.unspecified) == 1:
                result_df.loc[result_df['id']==id, 'vertebrae_fracture_unspecified'] = 1

    result_df.to_csv(save_path, index=False)
    print("Osseous CSV file created successfully.")