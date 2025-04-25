import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import json
import os

def calculate_organ_f1(pred_path, gt_path, output_path):
    """
    두 Excel 파일에서 F1 점수를 계산하고 각 장기별로 JSON 파일로 저장합니다.
    
    Args:
        pred_path (str): 예측 결과가 저장된 Excel 파일 경로
        gt_path (str): Ground Truth가 저장된 Excel 파일 경로
        output_path (str): 결과를 저장할 JSON 파일 경로
    """
    # 장기별 파일 정의
    organs = [
        'lung', 
        'large_airway', 
        'mediastinum', 
        'heart_and_vessel', 
        'abdomen', 
        'osseous_structure'
    ]
    
    # 결과를 저장할 딕셔너리
    results = {}
    
    # 전체 셀 단위 계산을 위한 리스트
    all_preds = []
    all_gts = []
    
    # 각 장기별로 F1 점수 계산
    for organ in organs:
        print(f"Processing {organ}...")
        
        # 파일 경로 설정
        pred_file = os.path.join(pred_path, f"{organ}.csv")
        gt_file = os.path.join(gt_path, f"{organ}_gt.csv")
        
        # 파일 존재 확인
        if not os.path.exists(pred_file) or not os.path.exists(gt_file):
            print(f"Warning: Files for {organ} not found. Skipping.")
            results[organ] = {"f1_score": None, "columns": []}
            continue
        
        # 데이터 로드
        pred_df = pd.read_csv(pred_file)
        gt_df = pd.read_csv(gt_file)
                
        # ID로 두 데이터프레임 정렬
        pred_df = pred_df.set_index('id')
        gt_df = gt_df.set_index('id')
        
        # 공통 ID만 사용
        common_ids = list(set(pred_df.index).intersection(set(gt_df.index)))
        if len(common_ids) == 0:
            print(f"Warning: No common IDs found for {organ}. Skipping.")
            results[organ] = {"f1_score": None, "columns": []}
            continue
        
        pred_df = pred_df.loc[common_ids]
        gt_df = gt_df.loc[common_ids]
        
        # 공통 칼럼 찾기 (첫 두 칼럼 제외: id와 report)
        pred_cols = set(pred_df.columns)
        gt_cols = set(gt_df.columns)
        
        # dataframe row 개수 일치하지 않으면 error
        if pred_df.shape[0] != gt_df.shape[0]:
            print(f"Warning: Row count mismatch for {organ}.")
            raise ValueError(f"Row count mismatch for {organ}.")
        
        # Column 확인
        if pred_cols != gt_cols:
            print(f"Warning: Column mismatch for {organ}. Skipping.")
            raise ValueError(f"Column mismatch for {organ}.")
        
        # id 칼럼 제외
        if 'id' in pred_cols:
            pred_cols.remove('id')
        if 'id' in gt_cols:
            gt_cols.remove('id')
        
        # report 칼럼 제외
        if f'{organ}_report' in pred_cols:
            pred_cols.remove(f'{organ}_report')
        if f'{organ}_report' in gt_cols:
            gt_cols.remove(f'{organ}_report')
            
        common_cols = list(pred_cols.intersection(gt_cols))
        
        if len(common_cols) == 0:
            print(f"Warning: No common columns found for {organ}. Skipping.")
            raise ValueError(f"No common columns found for {organ}.")
        
        # 각 공통 칼럼에 대해 F1 계산
        column_scores = {}
        
        for col in common_cols:
            try:
                # 데이터가 숫자인지 확인
                if pd.api.types.is_numeric_dtype(pred_df[col]) and pd.api.types.is_numeric_dtype(gt_df[col]):
                    # 결측값 처리
                    pred_col = pred_df[col].fillna(0).astype(int)
                    gt_col = gt_df[col].fillna(0).astype(int)
                    
                    # 전체 셀 단위 계산을 위해 모든 예측값과 실제값 저장
                    all_preds.extend(pred_col.tolist())
                    all_gts.extend(gt_col.tolist())
                    
                    # F1 계산
                    score = f1_score(gt_col, pred_col, zero_division=0)
                    column_scores[col] = float(score)  # NumPy 타입을 일반 float로 변환
                else:
                    print(f"Warning: Column {col} in {organ} is not numeric. Skipping.")
            except Exception as e:
                print(f"Error processing {col} in {organ}: {str(e)}")
        
        # 평균 F1 계산
        if column_scores:
            avg_f1 = sum(column_scores.values()) / len(column_scores)
            results[organ] = {
                "f1_score": float(avg_f1),  # NumPy 타입을 일반 float로 변환
                "columns": column_scores
            }
        else:
            results[organ] = {"f1_score": None, "columns": {}}
    
    # 이전 방식: 장기별 평균 F1 계산
    valid_f1_scores = [results[organ]["f1_score"] for organ in organs if results[organ]["f1_score"] is not None]
    if valid_f1_scores:
        results["organ_average"] = {"f1_score": float(sum(valid_f1_scores) / len(valid_f1_scores))}
    else:
        results["organ_average"] = {"f1_score": None}
    
    # 새로운 방식: 전체 셀 단위의 F1 계산
    if all_preds and all_gts:
        overall_f1 = f1_score(all_gts, all_preds, zero_division=0)
        results["Total"] = {"f1_score": float(overall_f1)}
    else:
        results["Total"] = {"f1_score": None}
    
    # 결과를 JSON으로 저장
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")
    
    return results