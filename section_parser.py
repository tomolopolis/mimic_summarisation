"""
A section parser module that parses a MIMIC-CXR report into multiple sections.

Code adapted from the MEDIQA Task3 shared task:
https://github.com/abachaa/MEDIQA2021/tree/main/Task3
and originally from the MIMIC-CXR GitHub repo at:
https://github.com/MIT-LCP/mimic-cxr/tree/master/txt.
"""

import re


def section_nurs_notes(text:str ):
    sections = []
    section_names = []

    start = None
    sec_names = ['assessment:', 'action:', 'response:', 'plan:']
    for idx, match in enumerate(re.compile('|'.join(sec_names), re.IGNORECASE).finditer(text)):
        if idx == 0:
            section_names.append('preamble')
            sections.append(text[0:match.start()])
        section_names.append(text[match.start():match.end()].lower().strip().replace(':', ''))
        if start:
            sections.append(text[start:match.start()])
        start = match.end()
    if start:
        sections.append(text[start:])
    return sections, section_names, []


def section_phys_texts(text: str, sec_names):
    sections = []
    section_names = []
    curr_sec_text_start = None
    curr_sec_text_end = None

    for idx, group in enumerate(re.compile('|'.join(sec_names), re.IGNORECASE).finditer(text)):
        if idx == 0:
            section_names.append('preamble')
            curr_sec_text_start = 0
        sec_name = text[group.start():group.end()].lower().strip().replace(':', '')
        if curr_sec_text_start is not None:
            curr_sec_text_end = group.start() - 1
            sections.append(text[curr_sec_text_start:curr_sec_text_end])
            curr_sec_text_start, curr_sec_text_end = None, None
        section_names.append(sec_name)
        curr_sec_text_start = group.end()
    if curr_sec_text_start:
        sections.append(text[curr_sec_text_start:])
    return sections, section_names, []


def section_physician_intensivist_note(text: str):
    return section_phys_texts(text, ['assessment and plan:', 'assessment and plan',
                                     'neurologic:', 'cardiovascular:', 'pulmonary:',
                                     'nutrition:', 'renal:', 'hematology:', 'endocrine:',
                                     'infectious disease:', 'wounds:', 'imaging:', 'fluids:',
                                     'consults:', 'billing diagnosis:', 'icu care'])


def section_physician_prog_text(text: str):
    return section_phys_texts(text, ['assessment and plan', 'nutrition:'])


def section_physician_attnd_prog_text(text: str):
    return section_phys_texts(text, ['cxr:', 'assessment and plan', 'nutrition:'])


def section_resp_care_shft_note(text: str):
    return section_phys_texts(text, ['ventilation assessment', 'plan',
                                     'reason for continuing current ventilatory support:',
                                     'respiratory care shift procedures'])



def section_echo_text(text: str):
    sections = []
    section_names = []
    curr_sec_text_start = None
    curr_sec_text_end = None

    for idx, group in enumerate(re.compile('\n([a-z]+):\n?', re.IGNORECASE).finditer(text)):
        if idx == 0:
            section_names.append('preamble')
            curr_sec_text_start = 0
        sec_name = text[group.start():group.end()].lower().strip().replace(':', '')
        if sec_name in _frequent_sections.keys():
            if curr_sec_text_start is not None:
                curr_sec_text_end = group.start() - 1
                sections.append(text[curr_sec_text_start:curr_sec_text_end])
                curr_sec_text_start, curr_sec_text_end = None, None
            section_names.append(_frequent_sections[sec_name])
            curr_sec_text_start = group.end()
    if curr_sec_text_start:
        sections.append(text[curr_sec_text_start:])
    return sections, section_names, []


def section_text(text):
    """Splits text into sections.

    Assumes text is in a radiology report format, e.g.:

        COMPARISON:  Chest radiograph dated XYZ.

        IMPRESSION:  ABC...

    Given text like this, it will output text from each section,
    where the section type is determined by the all caps header.

    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(
        r'\n ([A-Z ()/,-]+)\n?[:|;]\s?', re.DOTALL | re.IGNORECASE)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)

    section_names = normalize_section_names(section_names)

    # remove empty sections
    # this handles when the report starts with a finding-like statement
    #  .. but this statement is not a section, more like a report title
    #  e.g. p10/p10103318/s57408307
    #    CHEST, PA LATERAL:
    #
    #    INDICATION:   This is the actual section ....
    # it also helps when there are multiple findings sections
    # usually one is empty
    for i in reversed(range(len(section_names))):
        if section_names[i] in ('impression', 'findings'):
            if sections[i].strip() == '':
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    if ('impression' not in section_names) & ('findings' not in section_names):
        # create a new section for the final paragraph
        if '\n \n' in sections[-1]:
            sections.append('\n \n'.join(sections[-1].split('\n \n')[1:]))
            sections[-2] = sections[-2].split('\n \n')[0]
            section_names.append('last_paragraph')
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx


def normalize_section_names(section_names):
    # first, lower case all
    section_names = [s.lower().strip() for s in section_names]

    p_findings = [
        'chest',
        'portable',
        'pa and lateral',
        'lateral and pa',
        'ap and lateral',
        'lateral and ap',
        'frontal and',
        'two views',
        'frontal view',
        'pa view',
        'ap view',
        'one view',
        'lateral view',
        'bone window',
        'frontal upright',
        'frontal semi-upright',
        'ribs',
        'pa and lat'
    ]
    p_findings = re.compile('({})'.format('|'.join(p_findings)))

    main_sections = [
        'impression', 'findings', 'history', 'comparison',
        'addendum'
    ]
    for i, s in enumerate(section_names):
        if s in _frequent_sections:
            section_names[i] = _frequent_sections[s]
            continue

        main_flag = False
        for m in main_sections:
            if m in s:
                section_names[i] = m
                main_flag = True
                break
        if main_flag:
            continue

        m = p_findings.search(s)
        if m is not None:
            section_names[i] = 'findings'

        # if it looks like it is describing the entire study
        # it's equivalent to findings
        # group similar phrasings for impression

    return section_names


# the numbers of occurrences are wrong / different, as this is for MIMIC-III not CXR.
_frequent_sections = {
    "preamble": "preamble",  # 227885
    "impression": "impression",  # 187759
    "comparison": "comparison",  # 154647
    "indication": "indication",  # 153730
    "findings": "findings",  # 149842
    "examination": "examination",  # 94094
    "technique": "technique",  # 81402
    "history": "history",  # 45624
    "comparisons": "comparison",  # 8686
    "clinical history": "history",  # 7121
    "reason for examination": "indication",  # 5845
    "notification": "notification",  # 5749
    "reason for exam": "indication",  # 4430
    "clinical information": "history",  # 4024
    "exam": "examination",  # 3907
    "clinical indication": "indication",  # 1945
    "conclusion": "impression",  # 1802
    "conclusions": "impression",
    "concusion": "impression",
    "chest, two views": "findings",  # 1735
    "recommendation(s)": "recommendations",  # 1700
    "type of examination": "examination",  # 1678
    "reference exam": "comparison",  # 347
    "patient history": "history",  # 251
    "addendum": "addendum",  # 183
    "comparison exam": "comparison",  # 163
    "date": "date",  # 108
    "comment": "comment",  # 88
    "findings and impression": "impression",  # 87
    "wet read": "wet read",  # 83
    "comparison film": "comparison",  # 79
    "recommendations": "recommendations",  # 72
    "findings/impression": "impression",  # 47
    "pfi": "history",
    'recommendation': 'recommendations',
    'wetread': 'wet read',
    'summary': 'impression',
    'impresssion': 'impression',
    'impressio': 'impression',
    'ndication': 'impression',  # 1
    'impresson': 'impression',  # 2
    'imprression': 'impression',  # 1
    'imoression': 'impression',  # 1
    'impressoin': 'impression',  # 1
    'imprssion': 'impression',  # 1
    'impresion': 'impression',  # 1
    'imperssion': 'impression',  # 1
    'mpression': 'impression',  # 1
    'impession': 'impression',  # 3
    'findings/ impression': 'impression',  # ,1
    'finding': 'findings',  # ,8
    'findins': 'findings',
    'findindgs': 'findings',  # ,1
    'findgings': 'findings',  # ,1
    'findngs': 'findings',  # ,1
    'findnings': 'findings',  # ,1
    'finidngs': 'findings',  # ,2
    'idication': 'indication',  # ,1
    'reference findings': 'findings',  # ,1
    'comparision': 'comparison',  # ,2
    'comparsion': 'comparison',  # ,1
    'comparrison': 'comparison',  # ,1
    'comparisions': 'comparison'  # ,1
}

if __name__ == '__main__':
    text = """
 Glucose
   82
   83
   83
   97
   99
   Other labs: PT / PTT / INR:15.0/32.1/1.3, Lactic Acid:0.8 mmol/L,
   Ca:8.2 mg/dL, Mg:1.8 mg/dL, PO4:2.9 mg/dL
   Assessment and Plan
   INEFFECTIVE COPING, HYPOTENSION (NOT SHOCK), ANEMIA, ACUTE, SECONDARY
   TO BLOOD LOSS (HEMORRHAGE, BLEEDING), .H/O AIRWAY, INABILITY TO PROTECT
   (RISK FOR ASPIRATION, ALTERED GAG, AIRWAY CLEARANCE, COUGH), .H/O
   ALCOHOL ABUSE, .H/O CANCER (MALIGNANT NEOPLASM), ESOPHAGEAL,
   HEMOPTYSIS, PAIN CONTROL (ACUTE PAIN, CHRONIC PAIN), [**Last Name 121**] PROBLEM -
   ENTER DESCRIPTION IN COMMENTS
   Right Pharyngeal Carcinoma
   Assessment and Plan: 52yo M with endstage SCC of vallecula who
   presented w/ diffuse hemorrhage around neck.
   Neurologic: chronic pain; cont dilaudid GTT (reaching max allowed);
   methadone incro to 50TID with good effect, gabapentin; h/o sz cont
   keppra. ativan PRN
   Cardiovascular: no cardiovascular support
   Pulmonary: Trach, (Ventilator mode: CMV), Peep 12, RR 16, TV 450 FiO2
   60; unable to ween down PEEP.
   Gastrointestinal / Abdomen: TF's only if patient does not feel satiated
   Nutrition:
   Renal: Foley, Adequate UO
   Hematology: no longer following labs, no plans on transfusing.  CMO if
   acute bleed
   Endocrine: not following fingersticks
   Infectious Disease:
   Lines / Tubes / Drains: trach, g-tube, foley, PIVs
   Wounds:
   Imaging:
   Fluids: D5 1/2 NS, Potassium Chloride
   Consults: ENT, Paliative care
   Billing Diagnosis:
   ICU Care
   Nutrition:
   Glycemic Control:
   Lines:
   20 Gauge - [**2190-12-15**] 02:00 PM
   Prophylaxis:
   DVT: Boots
   Stress ulcer: PPI
   VAP bundle:
   Comments:
   Communication: Family meeting planning, ICU consent signed Comments:
   Planning [**12-17**] meeting ~1pm w/ palliative care and family to address IV
   fluid usage.
   Code status: DNR (do not resuscitate)
   Disposition: ICU
   Total time spent: 24 minutes
    """
    section_physician_intensivist_note(text)
