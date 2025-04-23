import dspy

class Mediastinum_Disease_Classifier(dspy.Module):
    def __init__(self):
        self.mediastinal_mass = dspy.ChainOfThought(Disease_Classifier_Mediastinal_Mass)
        self.lymphadenopathy = dspy.ChainOfThought(Disease_Classifier_Lymphadenopathy)
        self.esophageal_mass = dspy.ChainOfThought(Disease_Classifier_Esophageal_Mass)
        self.pneumomediastinum = dspy.ChainOfThought(Disease_Classifier_Pneumomediastinum)

    def forward(self, report):
        return {
            "Mediastinal_Mass": self.mediastinal_mass(report=report), 
            "Lymphadenopathy": self.lymphadenopathy(report=report),
            "Esophageal_Mass": self.esophageal_mass(report=report),
            "Pneumomediastinum" : self.pneumomediastinum(report=report)
        }
        
class Disease_Classifier_Mediastinal_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify whether a 'mediastinal mass' is explicitly mentioned in the report, and extract the relevant sentence(s) exactly as written.

    Instructions:
    1. Only include sentences that explicitly mention a 'mediastinal mass'.
    2. Do not include:
        - Mentions of 'lymphadenopathy', 'enlarged lymph nodes', or similar terms
        - Statements describing limited or non-diagnostic evaluation, such as:
            * "mediastinum cannot be evaluated"
            * "mediastinal structures not well visualized"
            * "suboptimal imaging"
            * "contrast was not given"
            * "limited evaluation"
        - General descriptions of mediastinal structures without reference to a mass
    3. Extract the exact sentence(s) from the report.
        - Do not paraphrase, summarize, or interpret.
        - Return only complete sentence(s) exactly as they appear.
    4. Set abnormality_presence = 1 if any valid mention of a mediastinal mass is found, otherwise set it to 0.

    Note:
    - Presence of a mediastinal mass must be explicitly stated. Indirect mentions or ambiguous references should be excluded.
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentence(s) mentioning mediastinal mass, or empty string if none found.")
    abnormality_presence: int = dspy.OutputField(desc="1 if mediastinal mass is present, else 0")
    


class Disease_Classifier_Lymphadenopathy(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify whether 'lymphadenopathy' is mentioned in the report, and extract the relevant sentence(s) exactly as written.
        - Relevant terms: lymphadenopathy, enlarged lymph nodes, lymph node enlargement 

    Instructions:
    1. Include sentences that:
        - Explicitly mention lymphadenopathy, enlarged lymph nodes, or lymph node enlargement.
        - Indirectly imply abnormality, such as:
            * Mention of lymph nodes in specific anatomical regions
            * Size measurements
            * Descriptions of prominence or abnormal appearance of lymph nodes
    2. Exclude sentences that:
        - Simply deny abnormality (e.g., "No enlarged lymph nodes") without mentioning anatomical location or size
        - Contain general negative findings without elaboration
    3. Extract the exact sentence(s) from the report.
        - Do not paraphrase, summarize, or interpret.
        - Return only complete sentence(s) exactly as they appear.
    4. Set abnormality_presence = 1 if any abnormal lymph node findings are present, otherwise set it to 0.

    Note:
    - Include sentences that describe lymph nodes in specific locations with size, even if labeled as stable or not pathological.
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentence(s) mentioning lymphadenopathy, or empty string if none found.")
    abnormality_presence: int = dspy.OutputField(desc="1 if lymphadenopathy is present, else 0")


class Disease_Classifier_Esophageal_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify whether an 'esophageal mass' is mentioned in the report, and extract the relevant sentence(s) exactly as written.

    Instructions:
    1. Include sentences that explicitly mention an esophageal mass or equivalent terms.
    2. Extract the complete sentence(s) that include the finding.
        - Do not modify, summarize, or interpret the sentence(s)
        - Return the full sentence(s) as they appear in the original report
    3. Set abnormality_presence = 1 if any valid mention of esophageal mass is found, otherwise set it to 0.

    Note:
    - Only include statements that clearly indicate the presence of an esophageal mass.
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentence(s) mentioning esophageal mass, or empty string if none found.")
    abnormality_presence: int = dspy.OutputField(desc="1 if esophageal mass is present, else 0")


class Disease_Classifier_Pneumomediastinum(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Identify whether 'pneumomediastinum' is mentioned in the report, and extract the relevant sentence(s) exactly as written.
        - relevant terms: penumomediastinum, (free) air in the mediastinum, mediastinal air, air densities in the medisastinum

    Instructions:
    1. Include sentences that explicitly mention pneumomediastinum or equivalent terms. 
    2. Extract the complete sentence(s) that include the finding.
        - Do not modify, summarize, or interpret the sentence(s).
        - Return the sentence(s) exactly as they appear in the original report.
    3. Set abnormality_presence = 1 if any valid mention of pneumomediastinum is found, otherwise set it to 0.

    Note:
    - Only include statements that clearly indicate the presence of pneumomediastinum.
    - Do not include statements indicating absence or uncertainty (e.g., "no pneumomediastinum", "pneumomediastinum not seen", "cannot exclude pneumomediastinum").
    """
    report: str = dspy.InputField(desc="Radiology report")

    lesion_sentence: str = dspy.OutputField(desc="Sentence(s) mentioning pneumomediastinum, or empty string if none found.")
    abnormality_presence: int = dspy.OutputField(desc="1 if pneumomediastinum is present, else 0")


# Locator
class Locator_Mediastinal_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Determine the anatomical location(s) of the mediastinal mass described in the report.

    Instructions:
    1. Based on the content of the report, identify whether a mediastinal mass is located in any of the following regions:
        - Anterior mediastinum
        - Middle mediastinum
        - Posterior mediastinum
        - Or if the location is unspecified
    2. Mark each region where a mediastinal mass is present with 1; otherwise, mark as 0.
    3. If the mass is described in multiple locations, set all corresponding fields to 1.
    4. If no mediastinal mass is found in the report, set all fields to 0.

    Note:
    - If a mediastinal mass is mentioned but the location is not specified, set `unspecified` to 1.
    - The report may contain multiple relevant sentences referring to different locations.
    """
    lesion_sentence: str = dspy.InputField(desc="Sentence(s) from the report describing mediastinal mass")

    anterior: int = dspy.OutputField(desc="1 if present in anterior mediastinum, else 0")
    middle: int = dspy.OutputField(desc="1 if present in middle mediastinum, else 0")
    posterior: int = dspy.OutputField(desc="1 if present in posterior mediastinum, else 0")
    unspecified: int = dspy.OutputField(desc="1 if location is unspecified, else 0")


class Locator_Lymphadenopathy(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Determine the anatomical location(s) of the lymphadenopathy described in the report.

    Instructions:
    1. Based on the content of the report, identify whether lymphadenopathy is present in any of the following regions (by anatomical name or station number):
        - Supraclavicular (Station 1)
        - Upper Paratracheal (Station 2)
        - Pre-vascular (Station 3A)
        - Pre-vertebral (Station 3P)
        - Lower Paratracheal (Station 4)
        - Subaortic (Station 5)
        - Paraaortic (Station 6)
        - Subcarinal (Station 7), relevant term: aorticopulmonary(aortopulmonary) window 
        - Paraesophageal (Station 8)
        - Hilar (Station 10)
        - Unspecified
    2. Mark each region with 1 if lymphadenopathy is present; otherwise, mark as 0.
    3. If multiple regions are mentioned, set all applicable fields to 1.
    4. If no lymphadenopathy is found in the report, set all fields to 0.

    Note:
    - If lymphadenopathy is mentioned but the location is not specified, set `unspecified` to 1.
    - The report may contain multiple relevant sentences referring to different regions.
    - If the report states only'paratrachea' without specifying **upper** or **lower**, set both `upper_paratracheal` and `lower_paratracheal` to 1. 
    """
    lesion_sentence: str = dspy.InputField(desc="Sentence(s) from the report describing lymphadenopathy")

    supraclavicular: int = dspy.OutputField(desc="1 if present in the supraclavicular region, else 0")
    upper_paratracheal: int = dspy.OutputField(desc="1 if present in the upper paratracheal region, else 0")
    prevascular: int = dspy.OutputField(desc="1 if present in the pre-vascular region, else 0")
    prevertebral: int = dspy.OutputField(desc="1 if present in the pre-vertebral region, else 0")
    lower_paratracheal: int = dspy.OutputField(desc="1 if present in the lower paratracheal region, else 0")
    subaortic: int = dspy.OutputField(desc="1 if present in the subaortic region, else 0")
    paraaortic: int = dspy.OutputField(desc="1 if present in the paraaortic region, else 0")
    subcarinal: int = dspy.OutputField(desc="1 if present in the subcarinal region, else 0")
    paraesophageal: int = dspy.OutputField(desc="1 if present in the paraesophageal region, else 0")
    hilar: int = dspy.OutputField(desc="1 if present in the hilar region, else 0")
    unspecified: int = dspy.OutputField(desc="1 if location is unspecified, else 0")

# Counter
class Counter_Esophageal_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report.

    Task:
    1. Determine whether an esophageal mass is described as a single occurrence or multiple occurrences in the report.

    Instructions:
    1. Analyze the content of the report to assess the number of distinct esophageal masses.
    2. Classify the count as one of the following:
        - Single: Exactly one esophageal mass is described.
        - Multiple: More than one esophageal mass is described.
    3. Set only one of the following fields to 1:
        - Set `single = 1` and `multiple = 0` if a single mass is present.
        - Set `multiple = 1` and `single = 0` if more than one mass is present.

    Note:
    - Do not infer or assume beyond what is clearly stated in the text.
    """
    lesion_sentence: str = dspy.InputField(desc="Radiology report describing esophageal mass")

    single: int = dspy.OutputField(desc="1 if a single esophageal mass is described, else 0")
    multiple: int = dspy.OutputField(desc="1 if multiple esophageal masses are described, else 0")
