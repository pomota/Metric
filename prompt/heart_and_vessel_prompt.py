import dspy

class Heart_and_Vessel_Disease_Classifier(dspy.Module):
    def __init__(self):
        self.classifiers = {
            'Aortic_Aneurysm': dspy.ChainOfThought(Disease_Classifier_Aortic_Aneurysm),
            'Aortic_Dilatation': dspy.ChainOfThought(Disease_Classifier_Aortic_Dilatation), 
            'Aortic_Dissection': dspy.ChainOfThought(Disease_Classifier_Aortic_Dissection),
            'Pulmonary_Artery_Enlargement': dspy.ChainOfThought(Disease_Classifier_Pulmonary_Artery_Enlargement),
            'Pulmonary_Embolism': dspy.ChainOfThought(Disease_Classifier_Pulmonary_Embolism),
            'Cardiomegaly': dspy.ChainOfThought(Disease_Classifier_Cardiomegaly),
            'Pericardial_Effusion': dspy.ChainOfThought(Disease_Classifier_Pericardial_Effusion),
            'Cardiac_Mass': dspy.ChainOfThought(Disease_Classifier_Cardiac_Mass),
            'Coronary_Artery_Wall_Calcification': dspy.ChainOfThought(Disease_Classifier_Coronary_Artery_Wall_Calcification),
            'Arterial_Calcification': dspy.ChainOfThought(Disease_Classifier_Arterial_Calcification)
        }

    def forward(self, report):
        return {
            disease: self.classifiers[disease](report=report)
            for disease in self.classifiers
        }
        
class Disease_Classifier_Aortic_Aneurysm(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'aortic aneurysm'.
        - Relevant terms: AA, TAA, AAA
        
    Instructions:
        1. Search for mentions of 'aortic aneurysm' or equivalent terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of pathology (e.g., "no aortic aneurysm")
            
        3. Set abnormality_presence = 1 if any sentences containing aortic aneurysm findings are present, otherwise set abnormality_presence = 0
        
    Note: 
        - Exclude search results related to enlargement of the heart or pulmonary arteries.
        - Simple statements about aortic enlargement or increased aortic width do not qualify as an aortic aneurysm. 
        - Only classify as an aortic aneurysm when the report explicitly uses the terms 'aneurysm' or 'AA' (Aortic Aneurysm).
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.
        
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'aortic aneurysm' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'aortic aneurysm' findings exist, 0 if lesion_sentence is empty")


class Disease_Classifier_Aortic_Dilatation(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'aortic dilatation'
        - meaning of dilatation : enlarged, increased vessels, vessels which are wider than normal. 
        - relevant terms of aorta: aortic arch, thoracic aorta.
        
    Instructions:
        1. Search for mentions of 'aortic dilatation' or equivalent terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'aortic dilatation' (e.g., "no aortic dilatation", "thoracic aorta diameter is normal")
            - Do not extract sentences if the description indicates an aortic aneurysm.
        
        3. Set abnormality_presence = 1 if any sentences containing aortic dilatation are present, otherwise set abnormality_presence = 0
    
    Note:
        - Exclude search results related to cardiomegaly(enlargement of the heart) or enlargement of pulmonary artery. Only include aortic dilation(enlargement of aorta).
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.
        
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'aortic dilatation' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'aortic dilatation' findings exist, 0 if lesion_sentence is empty")
    


class Disease_Classifier_Aortic_Dissection(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'aortic dissection'.
        - Relevant terms: aorta dissection
        
    Instructions:
        1. Search for mentions of 'aortic dissection' or equivalent terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'aortic dissection' (e.g., "no aortic dissection")
            
        3. Set abnormality_presence = 1 if any sentences containing 'aortic dissection' findings are present, otherwise set abnormality_presence = 0
        
    Note:
        - Aortic dissection indicates a tear in the aortic intima allows blood to create a false lumen between the layers of the aortic wall. Other aortic pathologies or lesions do not constitute aortic dissection.
        - Do not consider aortic dilatation (enlargement of the blood vessel) as aortic dissection.
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.
        
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'aortic dissection' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'aortic dissection' findings exist, 0 if lesion_sentence is empty")
    

class Disease_Classifier_Pulmonary_Artery_Enlargement(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'pulmonary artery enlargement'.
        - meaning of enlargement : enlarged, increased vessels, vessels which is wider than normal. 
        - relevant terms of pulmonary artery : pulmonary artery, pulmonary trunk
        
    Instructions:
        1. Search for mentions of 'pulmonary artery enlargement' or relevant terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'pulmonary artery enlargement' (e.g., "no pulmonary artery enlargement" or "pulmonary artery size is normal")
            
        3. Set abnormality_presence = 1 if any sentences containing 'pulmonary artery enlargement' findings are present, otherwise set abnormality_presence = 0
    
    Note:
        - Exclude search results related to cardiomegaly(enlargement of the heart) or aortic dilation. Only include enlargement of pulmonary artery.
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.

    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'pulmonary artery enlargement' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'pulmonary artery enlargement' findings exist, 0 if lesion_sentence is empty")
    
    
class Disease_Classifier_Pulmonary_Embolism(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'pulmonary embolism'.
        - Relevant terms: pulmonary thromboembolism, thromboembolism of the pulmonary artery, 'PE', 'PTE'
        
    Instructions:
        1. Search for mentions of 'pulmonary embolism' or relevant terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'pulmonary artery enlargement' (e.g., "no pulmonary embolism")
            - Do not extract any sentence which is ambiguous, inferential, or expresses an assumption regarding pulmonary embolism
            
        3. Set abnormality_presence = 1 if any sentences containing 'pulmonary embolism' findings are present, otherwise set abnormality_presence = 0
            
    Note:
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.

    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'pulmonary embolism' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'pulmonary embolism' findings exist, 0 if lesion_sentence is empty")
    
    
class Disease_Classifier_Cardiomegaly(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'cardiomegaly'.
        - Relevant terms: enlarged heart, increased heart size
        
    Instructions:
        1. Search for mentions of 'cardiomegaly' or relevant terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'cardiomegaly' (e.g. "no cardiomegaly" or "heart size is normal")
            
        3. Set abnormality_presence = 1 if any sentences containing 'cardiomegaly' findings are present, otherwise set abnormality_presence = 0
    
    Note:
        - Include findings of increased size of ventricle and atrium of heart. 
        - Statements indicating the absence of cardiomegaly do NOT count as positive findings.
        - Exclude search results related to artery enlargements. Only include enlargement of the heart.
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.

    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about if 'cardiomegaly' is present from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'cardiomegaly' findings exist, 0 if lesion_sentence is empty")
    
    
class Disease_Classifier_Pericardial_Effusion(dspy.Signature): 
    """
    You are a radiologist reviewing a radiology report
    - relevant terms : effusion of pericardium
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'pericardial effusion'.
        
    Instructions:
        1. Search for mentions of 'pericardial effusion' or relevant terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'pericardial effusion' (e.g. "no pericardial effusion thickening" or "no evidence of pericardial thickening")
            
        3. Set abnormality_presence = 1 if any sentences containing 'pericardial effusion' findings are present, otherwise set abnormality_presence = 0
        
    Note:
        - Expressions like "no pericardial effusion" or "pericardial effusion-thickening is not observed" should NOT be considered as pericardial effusion findings.
        - Pericardial thickening (increased thickness of the pericardium) should not be classified as pericardial effusion.
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.
        
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'pericardial effusion' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'pericardial effusion' findings exist, 0 if lesion_sentence is empty")
    

class Disease_Classifier_Cardiac_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'cardiac mass'.
        - Relevant terms: tumor, lesion, growth, nodule within the heart
        
    Instructions:
        1. Search for mentions of 'cardiac mass' or relevant terms.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear.
            - Do not extract any sentences that describe normal findings or absence of 'cardiac mass' (e.g. "no cardiac mass" or "heart is normal")
            
        3. Set abnormality_presence = 1 if any sentences containing 'cardiac mass' findings are present, otherwise set abnormality_presence = 0
        
    Note: 
        - Exclude general statements about heart size or cardiac silhouette.
        - Do not extract sentences related to mass of arteries, trunks, or blood vessels. Only include mass of heart.
        - If abnormality_presence is 0, set lesion_sentence to be an empty string.
        
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'cardiac mass' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'cardiac mass' findings exist, 0 if lesion_sentence is empty")
    

class Disease_Classifier_Coronary_Artery_Wall_Calcification(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'coronary artery wall calcification'.
        - Relevant terms: atherosclerotic plaques or atherosclerosis in coronary artery wall.
        - terms meaning of coronary arteries : RCA, LCA, LAD, LCx, wall of coronary artery
        
    Instructions:
        1. Search for mentions of 'coronary artery wall calcification' or relevant terms.
            - Include results mentioning atheroma plaques even if calcification is not explicitly stated.
        
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'coronary artery wall calcification' (e.g."no coronary artery wall calcification" or "coronary artery wall is normal")
            
        3. Set abnormality_presence = 1 if any sentences containing 'coronary artery wall calcification' findings are present, otherwise set abnormality_presence = 0
            
    Note:
    - Exclude search results related to arterial calcification or abdominal arterial calcification. Only include calcification of the coronary artery walls. 
    - Include atheroma plaques in coronary arteries even when calcification is not explicitly mentioned.
    - If abnormality_presence is 0, set lesion_sentence to be an empty string.

    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'coronary artery wall calcification' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'coronary artery wall calcification' findings exist, 0 if lesion_sentence is empty")
     

class Disease_Classifier_Arterial_Calcification(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to 'arterial calcification'.
        - Relevant terms: intimal calcification, vascular calcification, or atherosclerotic plaques in the aorta
        
    Instructions:
        1. Search for mentions of 'arterial calcification' or relevant terms.
            - Include results mentioning atheroma plaques even if calcification is not explicitly stated.
            
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
            - Do not cut off, modify, summarize, or interpret the extracted sentences - copy the entire sentence exactly as they appear
            - Do not extract any sentences that describe normal findings or absence of 'arterial calcification' (e.g."no arterial calcification")
            
        3. Set abnormality_presence = 1 if any sentences containing 'arterial calcification' are present, otherwise set abnormality_presence = 0
        
    Note:
    - Exclude search results related to coronary artery wall calcification. Only include calcification of the arteries.
    - Include atheroma plaques in arteries even when calcification is not explicitly mentioned.
    - If abnormality_presence is 0, set lesion_sentence to be an empty string.
    

    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences extracted about 'arterial calcification' in the report from the instruction #2 above, return empty string(no quotes, brackets or braces) if no relevant sentences were extracted")
    abnormality_presence: int = dspy.OutputField(desc="1 if abnormal 'arterial calcification' findings exist, 0 if lesion_sentence is empty")
    

class Locator_Pulmonary_Embolism(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Determine the location of the pulmonary embolism in the report.
    
    2. Abnormalities can be located in the following section.
        - Right pulmonary artery
        - Left pulmonary artery
        - main pulmonary trunk
        - unspecified
    
    Rules:
        - If the pulmonary embolism is not found, set all fields to 0.
        - If the pulmonary embolism is present, but the laterality is not specified in the report, it is unspecified.
    
    Note:
        - Do not infer or guess laterality if not clearly mentioned.
    """
    
    sentence: str = dspy.InputField(desc="Sentence describing an abnormality from the radiology report.")
    abnormality_class: str = dspy.InputField(desc="Abnormality to identify the location")
    
    right: int = dspy.OutputField(desc="1 if the abnormality is present in the right pulmonary artery, else 0")
    left: int = dspy.OutputField(desc="1 if the abnormality is present in the left pulmonary artery, else 0")
    main: int = dspy.OutputField(desc="1 if the location is in the main pulmonary trunk, else 0")
    unspecified: int = dspy.OutputField(desc="1 if the location is unspecified, else 0")
