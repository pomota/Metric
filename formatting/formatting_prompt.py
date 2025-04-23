import dspy

class Extract_LungParenchyma(dspy.Signature):
    """
    Task:
    You are given a medical report. Extract findings of the 'lung parenchyma', 'small airway' or 'pleural space' from the report.
    
    Instructions:
    1. Include only findings directly relevant to the 'lung parenchyma', 'small airway and ''pleural space'.
    Do not include keywords that indicate 'large airway' (e.g., trachea, bronchus, bronchi)
    
    Keywords that often indicate 'lung parenchyma', 'small airway' or 'pleural space' findings include:
    emphysema, nodule(s), consolidation, parenchymal distortion, ground-glass opacity (GGO), mass, fibrosis, infiltrates, atelectasis, scarring, linear opacity, bullae, lobe, interlobar...

    Keywords that often indicate 'small airway' findings include:
    bronchiectasis, bronchiole, bronchiolitis, bronchovascular ...
    
    Keywords that often indicate 'pleural space' findings include:
    pleural effusion, pleural thickening, subpleural, ...
    
    2. For each finding, include all associated descriptions provided in the report.
    
    3. Combine all descriptions referring to the same finding into a single entry, maintaining logical connections. (Ensure descriptions of the same finding are not divided into separate entries)
    Examples:
        Input:
        Lobulated mass, conglomerated with right paratracheal, hilar, interlobar lymph nodes.
        Invasion of right upper pulmonary vein, right upper lobar pulmonary arteries.

        Output:
        Lobulating mass, conglomerated with right paratracheal, hilar, interlobar lymph nodes that invades right upper pulmonary vein, right upper lobar pulmonary arteries.
        
    4. Use the exact wording from the given report without adding your thoughts or additional expressions.
    
    5. If no direct observations can be identified, write "No relevant findings."
    """
    report: str = dspy.InputField(desc="patient's medical report")
    sentences: str = dspy.OutputField(desc="sentences corresponding to lung parenchyma, small airway, or pleural space from the medical report")

class Extract_Airways(dspy.Signature):
    """
    Task:
    You are given a medical report. Extract findings of the 'Large airway' from the report.
    
    Instructions:
    1. Include only findings directly relevant to the 'Large airway'.
    Large airway indicates trachea, bronchus, bronchi.
    Keywords like Bronchioles, Bronchiectasis, Bronchiolitis, Bronchovascular are not included.
    Do not include terms that describe locations such as paratracheal or retrotracheal, as they pertain to positional descriptions rather than findings of the airway.
    
    2. Use the exact wording from the given report without adding your thoughts or additional expressions.
    
    3. If no direct observations can be identified, write "No relevant findings."    
    """
    report: str = dspy.InputField(desc="patient's medical report")
    sentences: str = dspy.OutputField(desc="sentences corresponding to Large airway from the medical report")

class Extract_Mediastinum(dspy.Signature):
    """
    Task:
    You are given a medical report. Extract findings of the 'mediastinum' from the report.
    
    Instructions:
    1. Include only findings directly relevant to the 'Mediastinum'.
    Keywords that often indicate [mediastinum] findings include:
    Esophagus, paratracheal, prevascular, paraaortic, subaortic, subcarinal, hilar, thymus, ...
    
    2. Use the exact wording from the given report without adding your thoughts or additional expressions.
  
    3. If no direct observations can be identified, write "No relevant findings."
    """
    report: str = dspy.InputField(desc="patient's medical report")
    sentences: str = dspy.OutputField(desc="sentences corresponding to the mediastinum from the medical report")

class Extract_HeartAndGreatVessels(dspy.Signature):
    """
    Task:
    You are given a medical report. Extract findings of the 'Heart and great vessels' from the report.
        
    Instructions:
    1. Include only findings directly relevant to the 'Heart and great vessels'.
    Do not include findings related to osseous structures. (e.g, bone, rib, fracture)
    Keywords that often indicate 'Heart and great vessels' findings include: 
    cardiomegaly, pericardial effusion, aortic aneurysm, pulmonary artery enlargement, vascular calcifications, pulmonary embolism, ..
    
    2. Use the exact wording from the given report without adding your thoughts or additional expressions.
    
    3. If no direct observations can be identified, write "No relevant findings."
    """
    report: str = dspy.InputField(desc="patient's medical report")
    sentences: str = dspy.OutputField(desc="sentences corresponding to the heart and great vessels from the medical report")

class Extract_Abdomen(dspy.Signature):
    """
    Task:
    You are given a medical report. Extract findings of the 'Abdomen' from the report.
        
    Instructions:
    1. Include only findings directly relevant to the 'Abdomen'.
    Keywords that often indicate 'Abdomen' findings include: 
    liver lesion, kidney, adrenal nodule, hepatic mass, splenomegaly, hiatal hernia ...

    2. Use the exact wording from the given report without adding your thoughts or additional expressions.

    3. If no direct observations can be identified, write "No relevant findings."
    """
    report: str = dspy.InputField(desc="patient's medical report")
    sentences: str = dspy.OutputField(desc="sentences corresponding to the abdomen from the medical report")

class Extract_OsseousStructures(dspy.Signature):
    """
    Task:
    You are given a medical report. Extract findings of the 'Osseous structures' from the report.
        
    Instructions:
    1. Include only findings directly relevant to the 'Osseous structures'.
    Keywords that often indicate 'Osseous structures' findings include: 
    rib lesion, lytic bone lesion, osteoblastic lesion, spinal involvement, fractures, bony metastasis, ...
    
    2. Use the exact wording from the given report without adding your thoughts or additional expressions.

    3. If no direct observations can be identified, write "No relevant findings."
    """
    report: str = dspy.InputField(desc="patient's medical report")
    sentences: str = dspy.OutputField(desc="sentences corresponding to the osseous structures from the medical report")