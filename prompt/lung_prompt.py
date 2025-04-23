# Locator prompt 수정
# Counter 밖으로 빼면서 prompt 수정

import dspy

class Lung_Disease_Classifier(dspy.Module):
    def __init__(self):
        # 여기서 안보고 싶은 병변만 주석처리
        self.classifiers = {
            'Pleural Effusion': dspy.ChainOfThought(Disease_Classifier_Pleural_Effusion),
            'Nodule': dspy.ChainOfThought(Disease_Classifier_Nodule),
            'Mass': dspy.ChainOfThought(Disease_Classifier_Mass),
            'Consolidation': dspy.ChainOfThought(Disease_Classifier_Consolidation),
            'Opacity': dspy.ChainOfThought(Disease_Classifier_Opacity),
            'Atelectasis': dspy.ChainOfThought(Disease_Classifier_Atelectasis),
            'Pneumothorax': dspy.ChainOfThought(Disease_Classifier_Pneumothorax),
            'Ground Glass Opacity': dspy.ChainOfThought(Disease_Classifier_Ground_Glass_Opacity),
            'Emphysema': dspy.ChainOfThought(Disease_Classifier_Emphysema),
            'Mosaic Attenuation': dspy.ChainOfThought(Disease_Classifier_Mosaic_Attenuation),
            'Bronchiectasis': dspy.ChainOfThought(Disease_Classifier_Bronchiectasis),
            'Interlobular Septal Thickening': dspy.ChainOfThought(Disease_Classifier_InterlobularSeptalThickening)
        }

    def forward(self, report):
        return {
            'lesion_sentence': {
                disease: self.classifiers[disease](report=report)['lesion_sentence']
                for disease in self.classifiers
            },
            'abnormality_presence': {
                disease: self.classifiers[disease](report=report)['abnormality_presence']
                for disease in self.classifiers
            }
        }

class Disease_Classifier_Nodule(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report. 
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Nodule].
        - Relevant terms: nodule, nodules, nodular lesion
    
    2. Only include findings in the lung or pleural space.
       
    Instructions: 
    1. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or add interpretation.
       - Include only complete sentences as they appear in the original report.
       - Do not extract findings that are absent. (e.g., "no nodule" or "no evidence of nodule")
    
    2. Set abnormality_presence = 1 if any [Nodule] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - [Mass] and [Nodule] are distinct entities. Ignore sentences referring only to masses or mass-like findings.
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Nodule] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Nodule] is present, else 0")

class Disease_Classifier_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Mass].
        - Relevant terms: mass, masses
    
    Instructions: 
    1. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or add interpretation. 
       - Include only complete sentences as they appear in the original report.
       - Do not extract findings that are absent. (e.g., "no mass" or "no evidence of mass")
    
    2. Set abnormality_presence = 1 if any [Mass] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - [Mass] and [Nodule] are distinct entities. Ignore sentences referring only to nodules.
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Mass] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Mass] is present, else 0")



class Disease_Classifier_Consolidation(dspy.Signature):
    """
    You are a radiologist reviewing a chest radiology report.
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Consolidation].
    
    Instructions: 
    1. If found, extract the complete sentence(s) containing these findings exactly as written in the original report
        - Do not modify, summarize, or interpret the extracted sentences - copy them exactly as they appear
        - Include only complete sentences as they appear in the original report.
        - Do not extract findings that are absent. (e.g., "no consolidation" or "no evidence of consolidation")
    
    2. Set abnormality_presence = 1 if any [Consolidation] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Consolidation] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Consolidation] is present, else 0")
    

class Disease_Classifier_Opacity(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report

    Task:
    1. Identify abnormal findings directly relevant to [Opacity] (excluding ground-glass opacity).
        - Relevant terms: opacity, opacities, focal opacity, patchy opacity, dense opacity, ill-defined opacity, mass-like opacity (excluding consolidation).

    Instructions: 
    1. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or interpret. 
       - Include only complete sentences as they appear in the original report.
       - Do not extract findings that are absent. (e.g., "no opacity" or "no evidence of opacity")
    
    2. Set abnormality_presence = 1 if any [Opacity] findings are present, otherwise set abnormality_presence = 0

    Note:
    - Exclude mentions of 'ground glass', 'GGOs', or similar terms indicating GGO.
    - Only include general or non-GGO opacity-related findings.
    - Do not extract [Fibrotic sequelae], [Atelectasis].
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning non-GGO [Opacities] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Opacity] is present, else 0")
    
    
class Disease_Classifier_Pleural_Effusion(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Pleural effusion].
    
    Instructions:            
    1. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or add interpretation. 
       - Include only complete sentences as they appear in the original report.
       - Do not extract findings that are absent. (e.g., "no pleural effusion" or "no evidence of pleural effusion")
       - Only extract findings that directly mention 'Pleural effusion' or its variants. 
    
    2. Set abnormality_presence = 1 if any [Pleural effusion] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - Do not extract [Pleural thickening] or [Pleural mass]
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Pleural effusion] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Pleural Effusion] is present, else 0")


class Disease_Classifier_Atelectasis(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Atelectasis].
    
    Instructions:  
    1. Extract the exact sentences from the report that mention these findings.
        - Do not summarize, rephrase, or add interpretation. 
        - Include only complete sentences as they appear in the original report.
        - Do not extract findings that are absent. (e.g., "no atelectasis" or "no evidence of atelectasis")
        - Only extract findings that directly mention 'Atelectasis' or its variants.
    
    2. Set abnormality_presence = 1 if any [Atelectasis] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Atelectasis] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Atelectasis] is present, else 0")
    
    
class Disease_Classifier_Pneumothorax(dspy.Signature):
    """
    You are a radiologist reviewing a chest radiology report.
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Pneumothorax].
    
    Instructions: 
    1. Extract the exact sentences from the report that mention these findings.
        - Do not summarize, rephrase, or add interpretation.
        - Include only complete sentences as they appear in the original report.
        - Do not extract findings that are absent. (e.g., "no pneumothorax" or "no evidence of pneumothorax")
    
    2. Set abnormality_presence = 1 if any [Pneumothorax] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Pneumothorax] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Pneumothorax] is present, else 0")
    

class Disease_Classifier_Ground_Glass_Opacity(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report

    Task:
    1. Identify abnormal findings directly relevant to [Ground Glass Opacity] (GGO).
        - Relevant terms: ground glass opacity, ground-glass opacities, ground-glass attenuation, GGOs, ground glass nodule (excluding subsolid nodule)

    Instructions: 
    1. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or interpret. 
       - Include only complete sentences as they appear in the original report.
       - Do not extract findings that are absent. (e.g., "no ground glass opacity" or "no evidence of ground glass opacity")
    
    2. Set abnormality_presence = 1 if any [Ground Glass Opacity] findings are present, otherwise set abnormality_presence = 0

    Note:
    - Exclude sentences that only refer to general opacities without ground glass features.
    - Focus specifically on GGO-like language that indicates early or subtle parenchymal abnormality.
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Ground Glass Opacities] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Ground Glass Opacity] is present, else 0")

    
class Disease_Classifier_Emphysema(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Emphysema].
        - Relevant terms: emphysema, emphysematous change, emphysematous aeration, emphysaematous appearance
    
    Instructions: 
    1. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or add interpretation. 
       - Include only complete sentences as they appear in the original report.
       - Do not extract findings that are absent. (e.g., "no emphysema" or "no evidence of emphysema")
    
    2. Set abnormality_presence = 1 if any [Emphysema] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Emphysema] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Emphysema] is present, else 0")


class Disease_Classifier_Mosaic_Attenuation(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Mosaic attenuation].
        - Relevant terms: mosaic attenuation, mosaic pattern, mosaic lung pattern, mosaic density differences
    
    Instructions:    
    1. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or add interpretation. 
       - Include only complete sentences as they appear in the original report.
       - Do not extract findings that are absent. (e.g., "no mosaic attenuation" or "no evidence of mosaic attenuation")
    
    2. Set abnormality_presence = 1 if any [Mosaic attenuation] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning [Mosaic attenuation] in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if [Mosaic Attenuation] is present, else 0")
    

class Disease_Classifier_Bronchiectasis(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify abnormal findings that are directly relevant to [Bronchiectasis].
        - Relevant terms: bronchiectasis, bronchiectatic changes, bronchiectatic appearance, bronchiectatic changes
    
    Instructions:
    1. Extract the exact sentences from the report that mention these findings.
        - Do not summarize, rephrase, or add interpretation.
        - Include only complete sentences as they appear in the original report.
        - Do not extract findings that are absent. (e.g., "no bronchiectasis" or "no evidence of bronchiectasis")
        - Only extract findings that directly mention 'Bronchiectasis' or its variants.
    
    2. Set abnormality_presence = 1 if any [Bronchiectasis] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - "Ectasia in bronchial structure" is [Bronchiectasis]
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """

    report = dspy.InputField(desc="Radiology report")
    
    lesion_sentence = dspy.OutputField(desc="Sentence mentioning [Bronchiectasis] in the report. If none, return an empty string.")
    abnormality_presence = dspy.OutputField(desc="Return 1 if [Bronchiectasis] is present, else return 0.")


class Disease_Classifier_InterlobularSeptalThickening(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task: 
    1. Identify abnormal findings that are directly relevant to [Interlobular Septal Thickening].
        - Relevant terms: interlobular septal thickening, septal lines, interstitial thickening
    
    Instructions:  
    1. Extract the exact sentences from the report that mention these findings.
        - Do not summarize, rephrase, or add interpretation.
        - Include only complete sentences as they appear in the original report.
        - Do not extract findings that are absent. (e.g., "no interlobular septal thickening" or "no evidence of interlobular septal thickening")
        - Only extract findings that directly mention 'Interlobular Septal Thickening' or its variants.
    
    2. Set abnormality_presence = 1 if any [Interlobular Septal Thickening] findings are present, otherwise set abnormality_presence = 0
    
    Note:
    - Not all thickening is [Interlobular septal thickening].
    - If the extracted lesion_sentence exists, then abnormality_presence is 1
    - If the extracted lesion_sentence is an empty string, then abnormality_presence is 0.
    """

    report = dspy.InputField(desc="Radiology report")

    lesion_sentence = dspy.OutputField(desc="Sentence mentioning [Interlobular septal thickening] in the report. If none, return an empty string.")
    abnormality_presence = dspy.OutputField(desc="Return 1 if [Interlobular septal thickening] is present, else return 0.")
    
    
# Locator
class Locator_RL(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Determine the location of the abnormalities in the report.
    
    2. Abnormalities can be located in the following section.
        - Right Lung
        - Left Lung
        - Both
        - Unspecified
    
    Instructions:
        - If the abnormality is present in both lungs, it is located both in the right lung and the left lung.
        - If the abnormality is present in the lung, but the laterality is not specified in the report, it is unspecified.
        - If the abnormality is not found set all fields to 0.
    
    Note:
    - Do not infer or guess laterality if not clearly mentioned.
    - When both a broader anatomical location (e.g., both lungs) and a specific lobe (e.g., LLL) are mentioned in the report, do not label only the specific lobe—include the broader region as well.
        Example:
        - abnormality_class="Consolidation"
        - report: "Consolidation in both lungs, especially in LLL."
        - Correct Labels: right=1, left=1, unspecified=0
    - When a secondary finding (e.g., GGO) is described in relation to another primary lesion (e.g., consolidation), such as "with adjacent GGO", infer the location of the secondary finding from the context and assign the same or neighboring anatomical region, unless clearly stated otherwise.
        Example:
        - abnormality_class="Ground Glass Opacity"
        - Report: "Consolidation in left lung, with adjacent GGO."
        - Correct Labels: right=0, left=1, unspecified=0
        
    Example 1: "lower lobes of both lungs" → right=1, left=1, unspecified=0
    Example 2: "in both lungs, especially in LLL" → right=1, left=1, unspecified=0
    """
    report: str = dspy.InputField(desc="Radiology report")
    abnormality_class: str = dspy.InputField(desc="Abnormality to identify the location")
    
    right: int = dspy.OutputField(desc="1 if the abnormality is present in the right lung, else 0")
    left: int = dspy.OutputField(desc="1 if the abnormality is present in the left lung, else 0")
    unspecified: int = dspy.OutputField(desc="1 if the location is unspecified, else 0")


class Locator_Left_Lobes(dspy.Signature):
    """
    You are a radiologist reviewing a chest radiology report.
    
    Task:
    1. Determine the location of the lobe of the abnormalities of the input in the report sentence.
    
    Instructions:
    1. Look for mentions of the specified abnormality in the report
    2. Determine if this abnormality is mentioned in relation to:
       - left upper lobe
       - left lower lobe
       - left lung (without specifying which lobe)
       
    Output rules:
    - left_upper_lobe = 1 if abnormality is specifically mentioned in left upper lobe, otherwise 0
    - left_lower_lobe = 1 if abnormality is specifically mentioned in left lower lobe, otherwise 0
    - unspecified = 1 if abnormality is mentioned without specifying which lobe, otherwise 0
    
    Note:
    - If the abnormality is not found set all fields to 0.
    - If report states the abnormality is in "both left lobes": set both left_upper_lobe=1 AND left_lower_lobe=1
    - Do not infer or guess laterality if not clearly mentioned.
    - If the abnormality is present in the lingular segment, it is located in the left upper lobe.
    - If the abnormality is present in the apical region or apex of the lung, it is located in the upper lobe.
    - When both a broader anatomical location (e.g., both lungs) and a specific lobe (e.g., LLL) are mentioned in the report, do not label only the specific lobe—include the broader region as well.
        Example:
        - abnormality_class="Consolidation"
        - sentence: "Consolidation in both lungs, especially in LLL."
        - Correct Labels: left_upper_lobe=0, left_lower_lobe=1, unspecified=1
    - When a secondary finding (e.g., GGO) is described in relation to another primary lesion (e.g., consolidation), such as "with adjacent GGO", infer the location of the secondary finding from the context and assign the same or neighboring anatomical region, unless clearly stated otherwise.
        Example:
        - abnormality_class="Ground Glass Opacity"
        - sentence: "Consolidation in LLL, with adjacent GGO."
        - Correct Labels: left_upper_lobe=0, left_lower_lobe=1, unspecified=0
    
    Example 1: "Consolidation is seen in the left lung" → left_upper_lobe=0, left_lower_lobe=0, unspecified=1
    Example 2: "Fibrosis in both lobes of the left lung" → left_upper_lobe=1, left_lower_lobe=1, unspecified=0
    Example 3: "Consolidation in the left upper lobe" → left_upper_lobe=1, left_lower_lobe=0, unspecified=0
    Example 4: "lower lobes of both lungs" → left_upper_lobe=0, left_lower_lobe=1, unspecified=0
    Example 5: "in both lungs, especially in LLL" → left_upper_lobe=0, left_lower_lobe=1, unspecified=1
    Example 5: "in both lungs, especially in RLL" → left_upper_lobe=0, left_lower_lobe=0, unspecified=1
    """
    sentence: str = dspy.InputField(desc="sentence that including lesions of the report")
    abnormality_class: str = dspy.InputField(desc="specific abnormality to find a location")
    
    left_upper_lobe: int = dspy.OutputField(desc="1 if abnormality is in left upper lobe, else 0")
    left_lower_lobe: int = dspy.OutputField(desc="1 if abnormality is in left lower lobe, else 0") 
    unspecified: int = dspy.OutputField(desc="1 if abnormality is in left lung but lobe not specified, else 0")

class Locator_Right_Lobes(dspy.Signature):
    """
    You are a radiologist reviewing a chest radiology report.
    
    Task:
    1. Determine the location of the lobe of the abnormalities of the input in the report sentence.
    
    Instructions:
    1. Look for mentions of the specified abnormality in the report
    2. Determine if this abnormality is mentioned in relation to:
       - right upper lobe
       - right middle lobe
       - right lower lobe
       - right lung (without specifying which lobe)
       
    Output rules:
    - right_upper_lobe = 1 if abnormality is specifically mentioned in right upper lobe, otherwise 0
    - right_middle_lobe = 1 if abnormality is specifically mentioned in right middle lobe, otherwise 0
    - right_lower_lobe = 1 if abnormality is specifically mentioned in right lower lobe, otherwise 0
    - unspecified = 1 if abnormality is mentioned without specifying which lobe, otherwise 0
    
    Note:
    - If the abnormality is not found set all fields to 0.
    - Do not infer or guess laterality if not clearly mentioned.
    - If the abnormality is present in the lingular segment, it is located in the left upper lobe.
    - If the abnormality is present in the apical region or apex of the lung, it is located in the upper lobe.
    - When both a broader anatomical location (e.g., both lungs) and a specific lobe (e.g., RLL) are mentioned in the report, do not label only the specific lobe—include the broader region as well.
        Example:
        - abnormality_class="Consolidation"
        - sentence: "Consolidation in both lungs, especially in RLL."
        - Correct Labels: right_upper_lobe=0, right_middle_lobe=0, right_lower_lobe=1, unspecified=1
    - When a secondary finding (e.g., GGO) is described in relation to another primary lesion (e.g., consolidation), such as "with adjacent GGO", infer the location of the secondary finding from the context and assign the same or neighboring anatomical region, unless clearly stated otherwise.
        Example:
        - abnormality_class="Ground Glass Opacity"
        - sentence: "Consolidation in RLL, with adjacent GGO."
        - Correct Labels: right_upper_lobe=0, right_middle_lobe=0, right_lower_lobe=1, unspecified=0
    
    Example 1: "Consolidation is seen in the right lung" → right_upper_lobe=0, right_middle_lobe=0, right_lower_lobe=0, unspecified=1
    Example 2: "Fibrosis in both lobes of the right lung" → right_upper_lobe=1, right_middle_lobe=0, right_lower_lobe=1, unspecified=0
    Example 3: "Consolidation in the right upper lobe" → right_upper_lobe=1, right_middle_lobe=0, right_lower_lobe=0, unspecified=0
    Example 4: "lower lobes of both lungs" → right_upper_lobe=0, right_middle_lobe=0, right_lower_lobe=1, unspecified=0
    Example 5: "in both lungs, especially in RLL" → right_upper_lobe=0, right_middle_lobe=0, right_lower_lobe=1, unspecified=1
    Example 5: "in both lungs, especially in LLL" → right_upper_lobe=0, right_middle_lobe=0, right_lower_lobe=0, unspecified=1
    """
    sentence: str = dspy.InputField(desc="sentence that including lesions of the report")
    abnormality_class: str = dspy.InputField(desc="specific abnormality to find a location")
    
    right_upper_lobe: int = dspy.OutputField(desc="1 if abnormality is in right upper lobe, else 0")
    right_middle_lobe: int = dspy.OutputField(desc="1 if abnormality is in right middle lobe, else 0")
    right_lower_lobe: int = dspy.OutputField(desc="1 if abnormality is in right lower lobe, else 0") 
    unspecified: int = dspy.OutputField(desc="1 if abnormality is in right lung but lobe not specified, else 0")

# Number
class Counter(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify and count the occurrences of a specified abnormality, abnormality_class in the report.
    2. Determine whether the abnormality is 'single' (one occurrence) or 'multiple' (more than one occurrence).
    
    Instructions:
    1. Look for mentions of the specified abnormality in the report.
    2. Classification criteria:
        - single = 1: Exactly ONE abnormality
        - multiple = 1: MORE THAN ONE abnormality
        - Mutually exclusive: If single = 1, then multiple = 0 (and vice versa)
        - One must be true: Either single = 1 OR multiple = 1
    
    Note:
    Count determination
   - Explicit count terms: "one", "two", "several", "multiple" 
   - Implicit indicators:
     * Singular forms without modifiers ("a nodule", "nodule") → single
     * Plural forms ("nodules") → multiple
     * Suffix indicators: "-s" ending indicates multiple (e.g., "masses", "lesions")
     * Quantity terms: "numerous", "few", "several", "multiple" → multiple
    """
    report: str = dspy.InputField(desc="Radiology report")
    abnormality_class: str = dspy.InputField(desc="Type of abnormality to count")
    
    single: int = dspy.OutputField(desc="1 if the abnormality count is single, else 0")
    multiple: int = dspy.OutputField(desc="1 if the abnormality count is multiple, else 0")