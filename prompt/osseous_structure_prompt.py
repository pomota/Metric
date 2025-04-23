import dspy

class Osseous_Structure_Disease_Classifier(dspy.Module):
    def __init__(self):
        self.classifiers = {
            'rib_fracture': dspy.ChainOfThought(Disease_Classifier_Rib_Fracture), ##수정
            'vertebrae_fracture': dspy.ChainOfThought(Disease_Classifier_Vertebrae_Fracture), ##수정
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
        
class Disease_Classifier_Rib_Fracture(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify whether the report contains a direct mention of 'rib fracture'.
                
    2. If there is a direct mention of 'rib fracture', extract the exact sentences from the report that mention rib fracture.
       - Do not summarize, rephrase, or add interpretation. 
       - Include only complete sentences as they appear in the original report.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning rib fracture in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if the abnormality is present in the report, else 0")


class Disease_Classifier_Vertebrae_Fracture(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify whether the report contains a direct mention of 'vertebrae fracture', 'compression', 'height loss', 'wedging'.
        - Relevant terms: vertebrae fracture, compression, height loss, wedging
        
    2. If there is a direct mention of 'vertebrae fracture', extract the exact sentences from the report that mention vertebrae fracture.
       - Do not summarize, rephrase, or add interpretation. 
       - Include only complete sentences as they appear in the original report.
    """
    report: str = dspy.InputField(desc="Radiology report")
    
    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning vertebrae fracture in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if the abnormality is present in the report, else 0")
    
    
# Locator
class Locator_Rib_Fracture(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
        1. Determine the location of rib fractures in the report. Location includes direction(right or left) and order(1st-12th) of the rib.
    
        2. Rib fractures can be located in the following section.
            - Right 1st rib
            - Right 2nd rib
            - Right 3rd rib
            - Right 4th rib
            - Right 5th rib
            - Right 6th rib
            - Right 7th rib
            - Right 8th rib
            - Right 9th rib
            - Right 10th rib
            - Right 11th rib
            - Right 12th rib
            - Left 1st rib
            - Left 2nd rib
            - Left 3rd rib
            - Left 4th rib
            - Left 5th rib
            - Left 6th rib
            - Left 7th rib
            - Left 8th rib
            - Left 9th rib
            - Left 10th rib
            - Left 11th rib
            - Left 12th rib
    
    Instructions: 
        - If rib fractures is not found set all fields to 0.
        - Binary output only. Do not output any rib numbers.
    
    Note:
        - Do not infer or guess laterality if not clearly mentioned.
    
    Examples:
        {report : 'In addition, fracture sequelae were observed in the lateral part of the 3-4th and 5th ribs on the left.',
        specified location : [left 3rd rib, left 4th rib, left 5th rib]},
        {report : 'A new fracture, which does not show separation, is observed in the left 5th rib.',
        specified location : [left 5th rib]}
    """
    lesion_sentence: str = dspy.InputField(desc="Sentence describing rib fractures from the radiology report.")
    
    right1: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 1st rib, otherwise return 0. Do not include any rib numbers.")
    right2: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 2nd rib, otherwise return 0. Do not include any rib numbers.")
    right3: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 3rd rib, otherwise return 0. Do not include any rib numbers.")
    right4: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 4th rib, otherwise return 0. Do not include any rib numbers.")
    right5: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 5th rib, otherwise return 0. Do not include any rib numbers.")
    right6: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 6th rib, otherwise return 0. Do not include any rib numbers.")
    right7: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 7th rib, otherwise return 0. Do not include any rib numbers.")
    right8: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 8th rib, otherwise return 0. Do not include any rib numbers.")
    right9: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 9th rib, otherwise return 0. Do not include any rib numbers.")
    right10: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 10th rib, otherwise return 0. Do not include any rib numbers.")
    right11: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 11th rib, otherwise return 0. Do not include any rib numbers.")
    right12: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the right 12th rib, otherwise return 0. Do not include any rib numbers.")
    left1: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 1st rib, otherwise return 0. Do not include any rib numbers.")
    left2: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 2nd rib, otherwise return 0. Do not include any rib numbers.")
    left3: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 3rd rib, otherwise return 0. Do not include any rib numbers.")
    left4: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 4th rib, otherwise return 0. Do not include any rib numbers.")
    left5: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 5th rib, otherwise return 0. Do not include any rib numbers.")
    left6: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 6th rib, otherwise return 0. Do not include any rib numbers.")
    left7: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 7th rib, otherwise return 0. Do not include any rib numbers.")
    left8: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 8th rib, otherwise return 0. Do not include any rib numbers.")
    left9: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 9th rib, otherwise return 0. Do not include any rib numbers.")
    left10: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 10th rib, otherwise return 0. Do not include any rib numbers.")
    left11: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 11th rib, otherwise return 0. Do not include any rib numbers.")
    left12: int = dspy.OutputField(desc="Binary output only: Return 1 if a rib fracture is specified in the left 12th rib, otherwise return 0. Do not include any rib numbers.")
    unspecified: int = dspy.OutputField(desc="Binary output only: Return 1 if the location is unspecified, otherwise return 0.")



class Locator_Vertebrae_Fracture(dspy.Signature):
    """
    You are a radiologist reviewing a chest radiology report.
    
    Task:
        1. Determine the location of vertebrae fractures of the input in the report sentence.
        2. Vertebrae fractures can be located in the following section.
            - C7 (7th cervical vertebra)
            - T1 (1st thoracic vertebra)
            - T2 (2nd thoracic vertebra)
            - T3 (3rd thoracic vertebra)
            - T4 (4th thoracic vertebra)
            - T5 (5th thoracic vertebra)
            - T6 (6th thoracic vertebra)
            - T7 (7th thoracic vertebra)
            - T8 (8th thoracic vertebra)
            - T9 (9th thoracic vertebra)
            - T10 (10th thoracic vertebra)
            - T11 (11th thoracic vertebra)
            - T12 (12th thoracic vertebra)
            - L1 (1st lumbar vertebra)
            - L2 (2nd lumbar vertebra)
            - L3 (3rd lumbar vertebra)
    
    Instructions:
        - If vertebrae fractures is not found set all fields to 0.
        - Binary output only. Do not output any rib numbers.
    
    Note:
        - Do not infer or guess laterality if not clearly mentioned.
    """
    lesion_sentence: str = dspy.InputField(desc="Sentence describing the vertebrae fracture from the radiology report.")
    
    C7: int = dspy.OutputField(desc="1 if the vertebral fracture is in C7 (7th cervical vertebra), else 0")
    T1: int = dspy.OutputField(desc="1 if the vertebral fracture is in T1 (1st thoracic vertebra), else 0")
    T2: int = dspy.OutputField(desc="1 if the vertebral fracture is in T2 (2nd thoracic vertebra), else 0")
    T3: int = dspy.OutputField(desc="1 if the vertebral fracture is in T3 (3rd thoracic vertebra), else 0")
    T4: int = dspy.OutputField(desc="1 if the vertebral fracture is in T4 (4th thoracic vertebra), else 0")
    T5: int = dspy.OutputField(desc="1 if the vertebral fracture is in T5 (5th thoracic vertebra), else 0")
    T6: int = dspy.OutputField(desc="1 if the vertebral fracture is in T6 (6th thoracic vertebra), else 0")
    T7: int = dspy.OutputField(desc="1 if the vertebral fracture is in T7 (7th thoracic vertebra), else 0")
    T8: int = dspy.OutputField(desc="1 if the vertebral fracture is in T8 (8th thoracic vertebra), else 0")
    T9: int = dspy.OutputField(desc="1 if the vertebral fracture is in T9 (9th thoracic vertebra), else 0")
    T10: int = dspy.OutputField(desc="1 if the vertebral fracture is in T10 (10th thoracic vertebra), else 0")
    T11: int = dspy.OutputField(desc="1 if the vertebral fracture is in T11 (11th thoracic vertebra), else 0")
    T12: int = dspy.OutputField(desc="1 if the vertebral fracture is in T12 (12th thoracic vertebra), else 0")
    L1: int = dspy.OutputField(desc="1 if the vertebral fracture is in L1 (1st lumbar vertebra), else 0")
    L2: int = dspy.OutputField(desc="1 if the vertebral fracture is in L2 (2nd lumbar vertebra), else 0")
    L3: int = dspy.OutputField(desc="1 if the vertebral fracture is in L3 (3rd lumbar vertebra), else 0")
    unspecified: int = dspy.OutputField(desc="1 if the location is unspecified, else 0")


# Rib fracture onset
class Onset_Rib_Fracture(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    Task: You are given a radiology report. Determine whether the onset of rib fracture at the given location is 'new' or 'old_healed'.

    Instructions:
    1. You will be given a sentence that describes the rib fracture from a radiology report.
    2. For given location of rib fracture, evaluate the onset of the rib fracture based on the following criteria:
       - new: Terms like " new", "just emerged" or similar descriptors are used
       - old_healed: Terms like "old", "healed", "previous" or similar descriptors are used
    3. For given two onset (new, old_healed), return:
       - 1 if the sentence indicates this onset
       - 0 if the sentence does not indicate this onset
    4. If the onset of the rib fracture is unspecified, return two onset (new, old_healed) fields 0 and return unspecified field 1.
    Notes:
    1. Only ONE of the two onset fields should be marked as 1, the others should be 0.
    """

    lesion_sentence = dspy.InputField(desc="Sentence describing the rib fracture from the radiology report.")
    location = dspy.InputField(desc="Location of the rib fracture from the radiology report. It can be given as 'unspecified'.")
    
    new = dspy.OutputField(desc="Return 1 if the onset of rib fracture is new, else return 0.")
    old_healed = dspy.OutputField(desc="Return 1 if the onset of rib fracture is old_healed, else return 0.")
    unspecified = dspy.OutputField(desc="Return 1 if the onset of rib fracture is unspecified, else return 0.")