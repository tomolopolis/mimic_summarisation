
# Section headers that appear after the brief hospital course.
# manually reveiwed next section headers.
next_section_headers = [
    '\n\nMedications on Admission:',
    '\n\nCONDITION ON DISCHARGE:',
    '\n\nDictated By:',
    '\n\nDISCHARGE MEDICATIONS:',
    '\n\nDISCHARGE DIAGNOSES:',
    '\n\nCONDITION AT DISCHARGE:',
    '\n\nDischarge Medications:',
    '\n\nDISCHARGE CONDITION:',
    '\n\nDISCHARGE STATUS:',
    '\n\nMEDICATIONS ON DISCHARGE:',
    '\n\nDISCHARGE DIAGNOSIS:',
    '\n\nDISPOSITION:',
    '\n\nDISCHARGE DISPOSITION:',
    '\n\nDischarge Disposition:',
    '\n\nTRANSITIONAL ISSUES:',
    '\n\nTransitional Issues:',
    '\n\nDISCHARGE INSTRUCTIONS:',
    '\n\nACTIVE ISSUES:',
    '\n\nActive Issues:',
    '\n\nPHYSICAL EXAMINATION ON DISCHARGE:',
    '\n\nCODE STATUS:',
    '\n\nTransitional issues:',
    '\n\nCV:',
    '\n\nFINAL DIAGNOSES:',
    '\n\nFEN:',
    '\n\nFOLLOW-UP:',
    '\n\nNOTE:',
    '\n\nCONDITION ON TRANSFER:',
    '\n\nFINAL DIAGNOSIS:',
    '\n\nPHYSICAL EXAMINATION:',
    '\n\nCode:',
    '\n\nLABORATORY DATA:',
    '\n\nActive Diagnoses:',
    '\n\nPROBLEM LIST:',
    '\n\nCODE:',
    '\n\nCHRONIC ISSUES:',
    '\n\nPAST MEDICAL HISTORY:',
    '\n\nID:',
    '\n\nHOSPITAL COURSE:',
    '\n\nPast Medical History:',
    '\n\nICU Course:',
    '\n\nTRANSITIONAL CARE ISSUES:',
    '\n\nActive issues:',
    '\n\nDISCHARGE PHYSICAL EXAMINATION:',
    '\n\nIMPRESSION:',
    '\n\nDISCHARGE PHYSICAL EXAMINATION:  Vital signs:',
    '\n\nFOLLOW UP:',
    '\n\nCONDITION AT TRANSFER:',
    '\n\nMICU course:',
    '\n\nCONDITION OF DISCHARGE:',
    '\n\nICU course:',
    '\n\nMICU Course:',
    '\n\nMEDICATIONS:',
    '\n\nPostoperative course was remarkable for the following:',
    '\n\nPENDING RESULTS:',
    '\n\nACTIVE ISSUES BY PROBLEM:',
    '\n\nMICU COURSE:',
    '\n\nCAUSE OF DEATH:',
    '\n\nACTIVE DIAGNOSES:',
    '\n\nOf note:',
    '\n\nPLAN:',
    '\n\nRECOMMENDATIONS AFTER DISCHARGE:',
    '\n\nCONDITION:',
    '\n\nACUTE ISSUES:',
    '\n\nPlan:',
    '\n\nGI:',
    '\n\nHospital course is reviewed below by problem:',
    '\n\nFINAL DISCHARGE DIAGNOSES:',
    '\n\nDIAGNOSES:',
    '\n\nACTIVE PROBLEMS:',
    '\n\nPROCEDURE:',
    '\n\nMEDICATIONS AT THE TIME OF DISCHARGE:',
    '\n\nDISCHARGE PLAN:',
    '\n\nPENDING LABS:',
    '\n\nDISCHARGE FOLLOWUP:',
    '\n\nChronic Issues:',
    '\n\nHospital course:',
    '\n\nComm:',
    '\n\nFOLLOW-UP INSTRUCTIONS:',
    '\n\nHospital Course:',
    '\n\nSURGICAL COURSE:',
    '\n\nLABORATORY DATA ON DISCHARGE:',
    '\n\nCode status:',
    '\n\nAddendum:',
    '\n\nACUTE DIAGNOSES:',
    '\n\nLABS ON DISCHARGE:',
    '\n\nTransitions of care:',
    '\n\nFluids, electrolytes and nutrition:',
    '\n\nDISCHARGE INSTRUCTIONS/FOLLOWUP:',
    '\n\nDIAGNOSIS:',
    '\n\nTRANSITIONAL CARE:',
    '\n\nCode Status:',
    '\n\nEvents:',
    '\n\nISSUES:',
    '\n\nFLOOR COURSE:',
    '\n\nFloor Course:',
    '\n\nTransitional:',
]

# Radiology report section headers, as listed from: https://github.com/MIT-LCP/mimic-cxr/blob/master/txt/section_parser.py
_frequent_sections = {
        "preamble": "preamble",  
        "impression": "impression",  
        "comparison": "comparison",  
        "indication": "indication",  
        "findings": "findings",  
        "examination": "examination",  
        "technique": "technique",  
        "history": "history",  
        "comparisons": "comparison",  
        "clinical history": "history",  
        "reason for examination": "indication",  
        "notification": "notification",  
        "reason for exam": "indication",  
        "clinical information": "history",  
        "exam": "examination",  
        "clinical indication": "indication",  
        "conclusion": "impression",  
        "chest, two views": "findings",  
        "recommendation(s)": "recommendations",  
        "type of examination": "examination",  
        "reference exam": "comparison",  
        "patient history": "history",  
        "addendum": "addendum",  
        "comparison exam": "comparison",  
        "date": "date",  
        "comment": "comment",  
        "findings and impression": "impression",  
        "wet read": "wet read",  
        "comparison film": "comparison",  
        "recommendations": "recommendations",  
        "findings/impression": "impression",  
        "pfi": "history",
        'recommendation': 'recommendations',
        'wetread': 'wet read',
        'ndication': 'impression',
        'impresson': 'impression',
        'imprression': 'impression',
        'imoression': 'impression',
        'impressoin': 'impression',
        'imprssion': 'impression',
        'impresion': 'impression',
        'imperssion': 'impression',
        'mpression': 'impression',
        'impession': 'impression',
        'findings/ impression': 'impression',  
        'finding': 'findings',  
        'findins': 'findings',
        'findindgs': 'findings',  
        'findgings': 'findings',  
        'findngs': 'findings',  
        'findnings': 'findings',  
        'finidngs': 'findings',  
        'idication': 'indication',  
        'reference findings': 'findings',  
        'comparision': 'comparison',  
        'comparsion': 'comparison',  
        'comparrison': 'comparison',  
        'comparisions': 'comparison'  
    }

impression_section_headers = [k for k,v in _frequent_sections.items() if v == 'impression']
