import os

# Special character replacement
SPECIAL_CHARS_MAPPING_TO_ASCII = {
    u'\u0091': '\'',
    u'\u0092': '\'',
    u'\u0093': '\"',
    u'\u0094': '-',
    u'\u0096': '-',
    u'\u0097': '-',
    '·': '-',
    '¨': '-',
    u'\u0095': '\n', 
}

ENCODE_STRINGS_DURING_PREPROCESSING = {
    # Use this to encode Dr. as =D= to prevent it from being split into Dr and .
    'Dr.': '@D@'
}

# Service mappings
# Map all services to uppercase long forms
SERVICE_MAPPING = {
    'MED': 'MEDICINE',
    'VSU': 'SURGERY',
    'OBS': 'OBSTETRICS/GYNECOLOGY',
    'ORT': 'ORTHOPAEDICS',
    'General Surgery': 'SURGERY',
    'Biologic': 'BIOLOGIC',
    'Biologic Service': 'BIOLOGIC',
    'GYN': 'OBSTETRICS/GYNECOLOGY',
    'Biologics': 'BIOLOGIC',
    'Neurology': 'NEUROLOGY',
    'ACS': 'SURGERY',
    'Biologics Service': 'BIOLOGIC',
    'NEURO': 'NEUROLOGY',
    'PSU': 'SURGERY',
    'TRA': 'SURGERY',
    'OP': 'SURGERY',
    'Neuromedicine': 'NEUROLOGY',
    'ENT': 'OTOLARYNGOLOGY',
    'OBSTERTRIC/GYNECOLOGY': 'OBSTETRICS/GYNECOLOGY',
    'OB service': 'OBSTETRICS/GYNECOLOGY',
    'Vascular Service': 'SURGERY',
    'OB-GYN': 'OBSTETRICS/GYNECOLOGY',
    'Vascular': 'SURGERY',
    'Surgical': 'SURGERY',
    'Ob-GYN': 'OBSTETRICS/GYNECOLOGY',
    'General surgery': 'SURGERY',
    'TRANSPLANT ': 'SURGERY',
    'ACS Service': 'SURGERY',
    'Thoracic Surgery Service': 'SURGERY',
    'Otolaryngology': 'OTOLARYNGOLOGY',
    'GU': 'UROLOGY',
    'CSU': 'SURGERY',
    'NME': 'NEUROLOGY',
    'BIOLOGICS': 'BIOLOGIC',
    'GENERAL SURGERY': 'SURGERY',
    'SURGICAL ONCOLOGY': 'SURGERY',
    'Surgical Oncology': 'SURGERY',
    '': 'UNKNOWN'
}

