import dspy

class Abdomen_Disease_Classifier(dspy.Module):
    def __init__(self):
        # 여기서 안보고 싶은 병변만 주석처리
        self.classifiers = {
            'kidney_cyst': dspy.ChainOfThought(Disease_Classifier_KidneyCyst),
            'liver_cyst': dspy.ChainOfThought(Disease_Classifier_LiverCyst),
            'adrenal_mass': dspy.ChainOfThought(Disease_Classifier_AdrenalMass),
            'gallstone': dspy.ChainOfThought(Disease_Classifier_Gallstone),
            'hiatal_hernia': dspy.ChainOfThought(Disease_Classifier_HiatalHernia),
            'pneumoperitoneum': dspy.ChainOfThought(Disease_Classifier_Pneumoperitoneum),
        }

    def forward(self, report):
        # Use dictionary comprehension to classify diseases
        return {
            disease: self.classifiers[disease](report=report)
            for disease in self.classifiers
        }

class Disease_Classifier_KidneyCyst(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify abnormal findings directly relevant to 'kidney cyst'.
        - Relevant terms: kidney cysts, renal cyst, renal cysts, simple cyst in the kidney, cortical cyst

    2. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or add interpretation. 
       - Include only complete sentences as they appear in the original report.
    
    Note:
    - Do not include cysts from other organs (e.g., liver cyst, ovarian cyst).
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning kidney cysts in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if the abnormality is present in the report, else 0")

class Disease_Classifier_LiverCyst(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify any abnormal findings that are directly relevant to a 'liver cyst' or cystic lesions in the liver.
        - This includes direct mentions and synonymous phrases such as:
          - liver cyst, hepatic cyst, hepatic cysts
          - hydatid cyst (if located in the liver)
          - simple hepatic cyst, cystic lesion in the liver, cystic hypodense lesion in the liver
          - cyst located in the right lobe, left lobe, or hepatic dome

    2. Extract the exact sentence(s) from the report that mention these findings.
        - Include only the full sentence(s) **as written** in the original report.
        - Do not summarize, paraphrase, interpret, or split fragments.

    Note:
    - Do not include non-hepatic cysts (e.g., kidney cysts).
    
    Examples:
    - "There is a 2 cm simple hepatic cyst in the right lobe." → Include this sentence.
    - "A lesion of approximately 86x66 mm in size, compatible with a stage 5 hydatid cyst, is located in the right hepatic dome." → Include this sentence.
    - "Simple cysts are noted in the kidneys and liver." → Include this sentence, but only the part that refers to the liver.
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning liver cysts in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if the abnormality is present in the report, else 0")

class Disease_Classifier_AdrenalMass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify abnormal findings directly relevant to 'adrenal mass'.
        - Relevant terms: adrenal masses, adrenal adenoma, adrenal nodule, adrenal lesion.

    2. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or interpret.
       - Include only complete sentences as they appear in the original report.

    Exclusion Criteria:
        - Do NOT include descriptions of adrenal gland shape, thickness, or contour (e.g., "slight thickening", "no space-occupying lesion").
        - Do NOT include incidental statements without clear mention of a mass or lesion.
        - Do NOT include findings that refer to other anatomical regions (e.g., kidney mass, retroperitoneal mass).
    Example:
    - "Slight thickening is observed in the left adrenal." → do not include
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning adrenal masses in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if the abnormality is present in the report, else 0")

class Disease_Classifier_Gallstone(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify abnormal findings directly relevant to 'gallbladder stone'.
        - Relevant terms: gallbladder stones, gallstones, cholelithiasis, gallbladder calculi, Calcified densities in the gallbladder, nodular sequela calcification of gallbladder

    2. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or interpret.
       - Include only complete sentences as they appear in the original report.

    Note:
    - Do not include bile duct stones (choledocholithiasis) unless clearly in gallbladder.
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning gallbladder stones in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if the abnormality is present in the report, else 0")

class Disease_Classifier_HiatalHernia(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify abnormal findings directly relevant to 'hiatal hernia'.
        - Relevant terms: hiatal hernia, hiatus hernia, Sliding type hiatal hernia, Type 1 hiatal hernia 

    2. Extract the exact sentences from the report that mention these findings.
       - Do not summarize, rephrase, or interpret.
       - Include only complete sentences as they appear in the original report.

    Note:
    - Do not include other types of hernia (e.g., inguinal, umbilical, incisional).
    - Ignore any hernias located outside the upper abdominal region.
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning hiatal hernia in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if the abnormality is present in the report, else 0")

class Disease_Classifier_Pneumoperitoneum(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify abnormal findings directly relevant to 'pneumoperitoneum'.
        - Relevant terms: pneumoperitoneum, free air in the abdomen, free intraperitoneal air, air under the diaphragm, intraperitoneal free air, subdiaphragmatic free air

    2. Extract the exact sentences from the report that mention these findings.
    - Do not summarize, rephrase, or interpret.
    - Include only complete sentences as they appear in the original report.

    Note:
    - Do not include intraluminal gas (normal bowel gas).
    - Focus on findings suggesting abnormal presence of air outside the bowel lumen.
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentences mentioning pneumoperitoneum in the report, empty string if none.")
    abnormality_presence: int = dspy.OutputField(desc="1 if pneumoperitoneum is present in the report, else 0")


# Locator
class Locator_Kidney_RL(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Determine the location of the kidney cyst in the report.
    
    2. The kidney cyst can be located in the following:
        - Right Kidney
        - Left Kidney
        - Both
        - Unspecified

    Rules:
        - If cysts are described in both kidneys, the location is both right and left.
        - If the report mentions a kidney cyst without specifying laterality, it is 'unspecified'.
        - If no kidney cyst is present in the report, set all fields to 0.

    Note:
        - Do not infer or guess laterality if not clearly mentioned.
        - Ignore non-kidney cysts even if they are adjacent to the kidney region. (e.g., liver cyst, ovarian cyst).
    """
    report: str = dspy.InputField(desc="Radiology report")
    abnormality_class: str = dspy.InputField(desc="Abnormality to identify the location")  # expected: "kidney cyst"
    
    right: int = dspy.OutputField(desc="1 if the kidney cyst is present in the right kidney, else 0")
    left: int = dspy.OutputField(desc="1 if the kidney cyst is present in the left kidney, else 0")
    unspecified: int = dspy.OutputField(desc="1 if the location is unspecified, else 0")


class Locator_adrenal_RL(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Determine the location of the adrenal mass in the report.
    
    2. The adrenal mass can be located in the following:
        - Right Adrenal Gland
        - Left Adrenal Gland
        - Both
        - Unspecified

    Rules:
        - If adrenal masses are found on both sides, mark both right and left.
        - If the adrenal mass is mentioned without side, it is 'unspecified'.
        - If no adrenal mass is found in the report, set all fields to 0.

    Note:
        - Do not infer or guess laterality if not clearly mentioned.
        - Do not confuse adrenal findings with other nearby structures (e.g., upper pole of kidney).
    """
    report: str = dspy.InputField(desc="Radiology report")
    abnormality_class: str = dspy.InputField(desc="Abnormality to identify the location")  # expected: "adrenal mass"
    
    right: int = dspy.OutputField(desc="1 if the adrenal mass is present in the right adrenal gland, else 0")
    left: int = dspy.OutputField(desc="1 if the adrenal mass is present in the left adrenal gland, else 0")
    unspecified: int = dspy.OutputField(desc="1 if the location is unspecified, else 0")
    
# Number
class Counter_Kidneycyst(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify and count the occurrences of a specified abnormality within a given anatomical location.
    2. Determine whether the abnormality is 'single' (one occurrence) or 'multiple' (more than one occurrence).
    
    Rules:
        - Single: Exactly one abnormality at the specified location.(Ex. A hypodense lesion, which was considered compatible with a cortical cyst is a single cyst)
        - Multiple: More than one abnormality at the specified location.
        - Output Exclusivity: If `single = 1`, then `multiple = 0`, and vice versa.

    Note:
        - Do not include abnormalities located outside the given anatomical location.
    """
    lesion_sentence: str = dspy.InputField(desc="Sentence describing kidney cyst(s) from the radiology report")
    
    single: int = dspy.OutputField(desc="1 if the abnormality count is single, else 0")
    multiple: int = dspy.OutputField(desc="1 if the abnormality count is multiple, else 0")
    
class Counter_Livercyst(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.
    
    Task:
    1. Identify and count the number of 'liver cysts', 'cystic lesions in liver' described in the report.
    2. Determine whether the finding represents a single liver cyst or multiple liver cysts.
    
    Definitions:
    - Single: Exactly one liver cyst is described.
    - Multiple: More than one liver cyst is described.
    - Stage ≠ Count: Do NOT confuse stages (e.g., "stage 5 hydatid cyst") with quantity. Staging refers to classification, not the number of cysts.
    - If the report mentions terms like "some of which are of cystic density", "a few cystic lesions", "several cysts", interpret these as **multiple** cysts.
    - Consider **plural hints** like "lesions", "some", "a few", or "cystic changes" as signs of **multiple** cysts unless it clearly states only one.

    Examples:
    - "A lesion of approximately 86x66 mm in size, compatible with a stage 5 hydatid cyst, is located in the right hepatic dome." -> `single = 1`, `multiple = 0`
    Special Notes:
    - "There are hypodense lesions in the liver parenchyma, some of which are of cystic density..." -> `single = 0`, `multiple = 1`
    
    Rules:
    - Only count abnormalities that are specifically described as liver cysts.
        - Example: If both kidney cysts and liver cysts are mentioned in the report, only count the liver cysts.
    - Classification criteria:
        - Single: Exactly one liver cyst is described.
        - Multiple: More than one liver cyst is described.
    - Output Exclusivity: If `single = 1`, then `multiple = 0`, and vice versa.
    """
    lesion_sentence: str = dspy.InputField(desc="Sentence describing liver cyst(s) from the radiology report")
    
    single: int = dspy.OutputField(desc="1 if the abnormality count is single, else 0")
    multiple: int = dspy.OutputField(desc="1 if the abnormality count is multiple, else 0")