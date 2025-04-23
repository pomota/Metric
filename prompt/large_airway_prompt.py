import dspy

class Large_Airway_Disease_Classifier(dspy.Module):
    def __init__(self):
        self.classifiers = {
            'tracheal_stenosis': dspy.ChainOfThought(Disease_Classifier_Tracheal_Stenosis),
            'endotracheal_mass': dspy.ChainOfThought(Disease_Classifier_Endotracheal_Mass),
            'endobronchial_mass': dspy.ChainOfThought(Disease_Classifier_Endobronchial_Mass)
        }

    def forward(self, report):
        return {
            disease: self.classifiers[disease](report=report)
            for disease in self.classifiers
        }
        
class Disease_Classifier_Tracheal_Stenosis(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
        1. Identify abnormal findings that are directly relevant to 'Tracheal Stenosis'.
            - Relevant terms: tracheal narrowing, tracheal stricture
    
    Instructions:
        1. Search for mentions of 'Tracheal Stenosis' or equivalent terms.
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report.
            - Do not modify, summarize, or interpret the extracted sentences - copy them exactly as they appear.
            - Include only complete sentences as they appear in the original report.
        3. Set abnormality_presence = 1 if any tracheal stenosis findings are present, otherwise set abnormality_presence = 0.
    
    Note:
        - Tracheal stenosis refers to narrowing of the trachea (windpipe) that causes respiratory problems.
    """

    report = dspy.InputField(desc="Radiology report")
    lesion_sentence = dspy.OutputField(desc="Sentence mentioning Tracheal Stenosis from the report. If none exist, return an empty string.")
    abnormality_presence = dspy.OutputField(desc="Return 1 if Tracheal Stenosis is present, else return 0.")
    
class Disease_Classifier_Endotracheal_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
        1. Identify abnormal findings that are directly relevant to 'Endotracheal Mass'.
            - Relevant terms: tracheal mass, tracheal lesion, tracheal tumor, tracheal nodule
    
    Instructions:
        1. Search for mentions of 'Endotracheal Mass' or equivalent terms.
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report.
            - Do not modify, summarize, or interpret the extracted sentences - copy them exactly as they appear.
            - Include only complete sentences as they appear in the original report.
        3. Set abnormality_presence = 1 if any endotracheal mass findings are present, otherwise set abnormality_presence = 0.
        4. Determine if there is a single mass or multiple masses mentioned.
    
    Note:
        - Endotracheal mass refers to an abnormal growth within the trachea (windpipe).
    """

    report = dspy.InputField(desc="Radiology report")
    lesion_sentence = dspy.OutputField(desc="sentence mentioning Endotracheal Mass from the report. If none exist, return an empty string.")
    abnormality_presence = dspy.OutputField(desc="Return 1 if Endotracheal Mass is present, else return 0.")
    mass_count_single = dspy.OutputField(desc="Return 1 if only one mass is mentioned, else return 0.")
    mass_count_multiple = dspy.OutputField(desc="Return 1 if more than one mass is clearly mentioned, else return 0.")


class Disease_Classifier_Endobronchial_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
        1. Identify abnormal findings that are directly relevant to 'Endobronchial Mass'.
            - Relevant terms: bronchial mass, bronchial lesion, bronchial tumor, bronchial nodule
    
    Instructions:
        1. Search for mentions of 'Endobronchial Mass' or equivalent terms.
        2. If found, extract the complete sentence(s) containing these findings exactly as written in the original report.
            - Do not modify, summarize, or interpret the extracted sentences - copy them exactly as they appear.
            - Include only complete sentences as they appear in the original report.
        3. Set abnormality_presence = 1 if any endobronchial mass findings are present, otherwise set abnormality_presence = 0.
        4. Determine the location of the mass (left main bronchus, right main bronchus, main bronchus, or unspecified).
        5. For mass count fields:
           - Set mass_count_single = 1 ONLY if exactly one mass is mentioned in total, regardless of location, else set it to 0.
           - Set mass_count_multiple = 1 if more than one mass is mentioned OR if there are masses in different locations, else set it to 0.
           - Base this count on the total number of masses, not on vague terms like "masses" or "lesions".
    
    Note:
        - Endobronchial mass refers to an abnormal growth within a bronchus (airway).
        - The mass count fields should align with the location information - if masses are found in multiple locations, 
          mass_count_multiple should be 1 even if each location has only one mass.
    """
    
    report = dspy.InputField(desc="Radiology report")
    lesion_sentence = dspy.OutputField(desc="sentence mentioning Endobronchial Mass from the report. If none exist, return an empty string.")
    abnormality_presence = dspy.OutputField(desc="Return 1 if any Endobronchial Mass is present, else return 0.")
    mass_count_single = dspy.OutputField(desc="Return 1 if exactly one mass is mentioned in total, else return 0.")
    mass_count_multiple = dspy.OutputField(desc="Return 1 if more than one mass is mentioned OR if masses are found in different locations, else return 0.")

class Locator_Endobronchial_Mass(dspy.Signature):
    """
    You are a radiologist reviewing a radiology report
    
    Task:
        1. Determine the precise location of an Endobronchial Mass mentioned in a specific sentence.
    
    Instructions:
        1. Analyze the provided sentence for location information of the mass.
        2. Determine if the mass is present in the left bronchus, right bronchus or if location is unspecified.
        3. Set the corresponding location field to 1 if the mass is present in that location, otherwise set it to 0.
    
    Note:
        - Be precise about anatomical locations mentioned in the sentence.
        - Only mark a location as present if it is explicitly mentioned or clearly implied.
    """
    
    lesion_sentence = dspy.InputField(desc="Sentence mentioning an Endobronchial Mass")
    abnormality_class = dspy.InputField(desc="Type of abnormality being analyzed")

    left_main = dspy.OutputField(desc="Return 1 if mass is present in the left bronchus, else return 0.")
    right_main = dspy.OutputField(desc="Return 1 if mass is present in the right bronchus, else return 0.")
    unspecified = dspy.OutputField(desc="Return 1 if location is unspecified, else return 0.")