import itertools
import re
from src.preprocess.constants import *

# Globally used regexes
re_whitespace = re.compile(r'\s+', re.MULTILINE)
re_multiple_whitespace = re.compile(r'  +', re.MULTILINE)
re_paragraph = re.compile(r'\n{2,}', re.MULTILINE)
re_line_punctuation = re.compile(r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|\||~|»|«|“|”|-|_)+$', re.MULTILINE)
re_line_punctuation_wo_fs = re.compile(r'^(?:!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|\||~|»|«|“|”|-|_)+$', re.MULTILINE)
re_line_punctuation_wo_underscore = re.compile(r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|`|\{|\||\}|\||~|»|«|“|”|-)+$', re.MULTILINE)
re_ds_punctuation_wo_underscore = re.compile(r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|`|\{|\||\}|\||~|»|«|“|”|-)+')
re_fullstop = re.compile(r'^(?:\.)+$', re.MULTILINE)
re_newline_in_text = re.compile(r'(?<=\w)\n(?=\w)', re.MULTILINE)
re_incomplete_sentence_at_end = re.compile(r'(?<=\.)[^\.]+$', re.DOTALL)
re_item_element  = r'(?:-|\. |\*|•|\d+ |\d+\.|\d\)|\(\d+\)|\d\)\.|o |# )'
re_heading_general = r'[^\.\:\n]*(?::\n{1,2}|\?\n{1,2}|[^,]\n)'

# Itemize elements identified in mimic notes
ITEMIZE_ELEMENTS = [r'-', r'\. ', r'\*', r'•', r'\d+ ', r'\d+\.', r'\d\)', r'\(\d+\)', r'\d\)\.', r'o ', r'# ']

# determiend common patterns at the beginning of summaries that contain no information and should be removed.
# manualy inspected 1000 summaries and found the following patterns:
UNNECESSARY_SUMMARY_PREFIXES = {
    'template separator': re.compile(r'^={5,40}', re.MULTILINE),
    'template heading': re.compile(r'\A(?:Patient |CCU )?Discharge (?:Instructions|Worksheet):?\s*', re.IGNORECASE|re.DOTALL),
    'salutations': re.compile(r'\A(?:___,|(?:Dear|Hello|Hi|Ms|Mrs|Miss|Mr|Dr)(?: Ms| Mrs| Miss| Mr| Dr)?\.{0,1} (?:___)?(?: and family| family)?(?:,|\.|:|;| ){0,3}|)\s*', re.IGNORECASE),
    # allow up to one sentences (.,!,:,;) before thank you and two before pleasure (more specific) and remove until end of sentence.
    'thank you': re.compile(r'\A(?:[^\.!:;]*\.){0,1}[^\.!:;]*thank you[^\.!:;]*(?:\.|!|:|;)\s*', re.IGNORECASE|re.DOTALL),
    'pleasure': re.compile(r'\A(?:[^\.!:;]*\.){0,2}[^\.!:;]*(?:pleasure|priviledge|privilege)[^\.!:;]*(?:\.|!|:|;)\s*', re.IGNORECASE|re.DOTALL),
}

# focus on occurences with typical headings followed by a dashed list, leave the other as they are
WHY_WHAT_NEXT_HEADINGS = (r'^-{0,4}[^\S\r\n]{0,4}_{0,4}[^\S\r\n]{0,4}'  # optional start via -- ___ ...
                          r'(?:why .* admitted|why .* hospital|what brought .* hospital|why .* here|where .* hospital|why .* hospitalized|'  # first section
                          r'what was done|what .* hospital|was I .* hospital|when you .* hospital|what .* here|what .* admitted|what .* for you|what .* hospitalization|what .* stay|what happened .* ___|while .* here|'  # second section
                          r'what should .* next|what should .* hospital|what .* for me|when .* leave|what .* leave|what .* home|when .* home|what should .* leaving|what .* to do|'  # third section
                          r'when .* hospital|when .* come back|what .* discharge|what .* discharged)'
                          r'(?:\?|:|\?:)?\n{1,2}')
# add dash to ensure that item follows
WHY_WHAT_NEXT_HEADINGS_DASHED_LIST = re.compile(WHY_WHAT_NEXT_HEADINGS + r'-', re.MULTILINE|re.IGNORECASE)

# These are some common suffixes of 'you' refering to the patient and allowing to replace likely anonymization of '___' with 'you'
YOU_SUFFIXES = ['were admitted', 'were here', 'were followed', 'were started', 'were found', 'were maintained', 'were able', 'were seen',
                'were treated', 'were given', 'were told', 'were advised', 'were asked', 'were instructed', 'were recommended',
                'were initially evaluated',  'were hospitalized', 'were complaining', 'were discharged', 'were also', 'were at',
                'will not need', 'will need to follow', 'will need to', 'will start this'
                'should hear', 'should follow',
                'have recovered', 'have healed',
                'are now ready', 'unfortunately developed', 'had chest pain', 'suffered', 'hit your head', 'vomited', 'can expect to see', 'tolerated the procedure']
SIMPLE_DEIDENTIFICATION_PATTERNS = [
    ('You ', re.compile(r'(?:^|\. )___ (?=' + '|'.join(YOU_SUFFIXES) + r')', re.MULTILINE|re.IGNORECASE)),
    (' you ', re.compile(r'(?!' + ENCODE_STRINGS_DURING_PREPROCESSING['Dr.'] + ') ___ (?=' + '|'.join(YOU_SUFFIXES) + r')', re.MULTILINE|re.IGNORECASE)),  # Prevent that Dr. ___ is replaced
    (' you', re.compile(r'(?:(?<=giving)|(?<=giving thank)|(?<=giving we wish)|(?<=giving scheduled)|(?<=giving will call)|(?<=we assessed)) ___', re.MULTILINE|re.IGNORECASE)), 
    (' your ', re.compile(r' ___ (?=discharge|admission)', re.MULTILINE|re.IGNORECASE)),
    (' your ', re.compile(r'(?=directs all the other parts of|the brain is the part of|see occasional blood in) ___ ', re.MULTILINE|re.IGNORECASE)),  # from neurology stroke / urology template
]


# Identified patterns that mark the end of a summary with no relevant information following it.
def create_heading_rs(heading):
    # Either delimited by : or starting and ending with newline
    return [heading + r':', r'(?:^|\n)' + heading + '\n']
SUFFIXES_DICT = {
    # follow-up
    "followup headings": create_heading_rs(r'follow(?:-| ||)(?:up)? instructions'),
    "followup sentences": [r'(?:you should|you have|you will|please)[^\.]{0,50} follow(?:-| ||)(?:up)?',
                           r'(?:call|see|visit|attend)[^\.]{0,200} follow(?:-| ||)(?:up)?',
                           r'follow(?:-| ||)(?:up)? with[^\.]{0,50} (?:primary care|pcp|doctor|neurologist|cardiologist)',
                           r'you will [^\.]{10,50} (?:primary care|pcp|doctor|neurologist|cardiologist)',
                           r'The number for [^\.]{10,200} is listed below'], 
    # discharge
    "discharge headings": create_heading_rs(r'discharge instructions') + create_heading_rs(r'[^\.]{0,200} surgery discharge instructions'),
    "discharge sentences": [r'Please follow [^\.]{0,30}discharge instructions',
                            r'(?:cleared|ready for)[^\.]{0,50} discharge',
                            r'(?:are|were|being|will be) (?:discharge|sending)[^\.]{0,200} (?:home|rehab|facility|assisted|house)',
                            r'(?:note|take)[^\.]{0,100} discharge (?:instruction|paperwork)',
                            r'Below are your discharge instructions regarding'],
    # farewell
    "farewell pleasure": [r'It [^\.]{3,20} pleasure', r'was a pleasure'],
    "farewell priviledge": [r'It [^\.]{3,20} priviled?ge'],
    "farewell wish you": [r'wish(?:ing)? you', r'Best wishes', r'wish(?:ing)? [^\.]{0,20} luck'],
    "farewell general": [r'Sincerely', r'Warm regards', r'Thank you', r'Your[^\.]{0,10} care team', r'Your[^\.]{0,10} (?:doctor|PCP)'],

    # activity
    "activity headings": create_heading_rs(r'Activity') + create_heading_rs(r'Activity and [^\.]{4,20}'),
    # ama
    "ama sentences": [r'You [^\.]{0,60}decided to leave the hospital'],
    # appointments
    "appointments sentences": [r'(?:keep|follow|attend|go to|continue)[^\.]{1,100} (?:appointment|follow(?:-| ||)up)',
                               r'(?:appointment|follow(?:-| ||)up)[^\.]{1,100} (?:arranged|scheduled|made)',
                               r'(?:contact|call|in touch)[^\.]{1,100} (?:appointment|follow(?:-| ||)up)',
                               r'have[^\.]{0,100} (?:appointments?|follow(?:-| ||)up) with ',
                               r'see[^\.]{0,100} (?:appointments?|follow(?:-| ||)up) below',
                               r'provide[^\.]{0,100} phone number',
                               r'getting an appointment for you'],
    # case manager
    "case manager sentences": [r'contact[^\.]{0,100} case manager',
                               r'case manager[^\.]{0,100} (?:contact|call|in touch|give|arrange|schedule|make)'],
    # diet
    "diet headings": create_heading_rs(r'Diet') + create_heading_rs(r'Diet and [^\.]{4,20}'),
    # forward info
    "forward info sentences": [r'forward[^\.]{0,100} (?:information|info|paper(?: |-||)work)'],
    # instructions
    "instructions sentences": [r'Please (?:review|follow|check)[^\.]{1,100} instructions?',
                               r'should discuss this further with '],
    # medication
    "medication headings": create_heading_rs(r'(?:medications?|medicines?|antibiotics?|pills?)') +\
                           create_heading_rs(r'(?:medications?|medicines?|antibiotics?|pills?) ?(?:changes|list|as follows|on discharge|for [^\.]{0,80})') +\
                           create_heading_rs(r'(?:take|administer|give|prescribe|order|direct|start|continue)[^\.]{0,100} doses') +\
                           create_heading_rs(r'schedule for[^\.]{0,100}') +\
                           [r'(?:take|administer|give|prescribe|order|direct|start|continue)[^\.]{0,50} (?:medications?|medicines?|antibiotics?|pills?)[^\.]{0,100} (?:prescribe|list|as follows)'],
    "medication sentences": [r'(?:following|not make|not make any|not make a|no) change[^\.]{0,100} (?:medications?|medicines?|antibiotics?|pills?)',
                             r'(?:medications?|medicines?|antibiotics?|pills?)[^\.]{0,100} (?:prescribed|directed|ordered|listed below|change)',
                             r'(?:continue|resume|take)[^\.]{0,100} (?:all|other|your)[^\.]{0,50} (?:medications?|medicines?|antibiotics?|pills?)',
                             r'see[^\.]{0,100} list[^\.]{0,100} (?:medications?|medicines?|antibiotics?|pills?)',
                             r'were given[^\.]{0,50} (?:presecription|prescription)'],
    "medication items": [r'^(?:please)? (?:start|stop|continue) take'],
    # questions
    "questions sentences": [r'call [^\.]{1,200} (?:questions|question|concerns|concern|before leave)',
                            r'If [^\.]{1,200} (?:questions|question|concerns|concern)',
                            r'Please do not hesitate to contact us'],
    # home
    "home sentences": [r'(?:ready|when|safe)[^\.]{0,30} home'],
    # surgery or procedure
    "surgery procedure headings": create_heading_rs(r'Surgery[^\.]{0,10}Procedure') + create_heading_rs(r'Surgery') + create_heading_rs(r'Procedure') +
                                  create_heading_rs(r'Your Surgery') + create_heading_rs(r'Your Procedure') +
                                  create_heading_rs(r'Recent Surgery') + create_heading_rs(r'Recent Procedure'),
    # warning signs
    "warning signs sentences": [r'please seek medical (?:care|attention)',
                                r'to[^\.]{0,100} (?:ED(?:\.|,|;| )|ER(?:\.|,|;| )|Emergency Department|Emergency Room)',  # Required seperation after ED/ER 
                                r'(?:call|contact|experience|develop) [^\.]{1,200} following',
                                r'(?:call|contact)[^\.]{0,100} (?:develop|experience|concerning symptom|if weight|weight goes|doctor|physician|surgeon|provider|nurse|clinic|office|neurologist|cardiologist|hospital)',
                                r'Please (?:call|contact|seek)[^\.]{0,200} if',
                                r'If[^\.]{0,100} (?:develop|experience|concerning symptoms|worse)'],
    # wound care
    "wound care headings": create_heading_rs(r'Wound Care') + create_heading_rs(r'Wound Care Instructions?'),
    "wound care sentences": [r'GENERAL INSTRUCTIONS WOUND CARE You or a family member should inspect',
                             r'GENERAL INSTRUCTIONS WOUND CARE\nYou or a family member should inspect',
                             r'Please shower daily including washing incisions gently with mild soap[^\.]{0,10} no baths or swimming[^\.]{0,10} and look at your incisions',
                             r'wash incisions gently with mild soap[^\.]{0,10} no baths or swimming[^\.]{0,10} look at your incisions daily',
                             r'Do not smoke\. No pulling up, lifting more than 10 lbs\., or excessive bending or twisting\.',
                             r'Have a friend/family member check your incision daily for signs of infection'],

    # general instructions
    "other headings": list(itertools.chain(*[create_heading_rs(h) for h in [r'Anticoagulation', r'Pain control', r'Prevena dressing instructions',
                                                                            r'your bowels', r'Dressings', r'Pain management', r'Incision care',
                                                                            r'What to expect', r'orthopaedic surgery', r'Physical Therapy', r'Treatment Frequency',
                                                                            r'IMPORTANT PATIENT DETAILS', r'IMPORTANT PATIENT DETAILS 1\.']])) +\
                      create_heading_rs(r'Please see below[^\.]{1,50} hospitalization') +\
                      create_heading_rs(r'[^\.]{0,50} in the hospital we') +\
                      [r'CRITICAL THAT YOU QUIT SMOKING'],

    # Added after t-sne analysis: Picked outlying clusters with at least 5 examples and 80% rouge4 performance
    # For each template selected 5 examples with very high rouge4 scores
    # Stroke template
    "stroke template sentences": [r'a condition (?:where|in which) a blood vessel providing oxygen and nutrients to the brain (?:is blocked|bleed)',
                                  r'The brain is the part of your body that controls? and directs all the other parts of your body',
                                  r'damage to the brain[^\.]{0,200} can result in a variety of symptoms',
                                  r'can have many different causes, so we assessed you for medical conditions',
                                  r'In order to prevent future strokes,? we plan to modify those risk factors'],
    "stone template sentences": [r'You can expect to see occasional blood in your urine and to possibly experience some urgency and frequency',
                                 r'You can expect to see blood in your urine for at least 1 week and to experience some pain with urination, urgency and frequency',
                                 r'The kidney stone may or may not [^\.]{0,30} AND\/or there may fragments\/others still in the process of passing',
                                 r'You may experiences? some pain associated with spasm? of your ureter'],
    "aortic graft template sentences": [r'You tolerated the procedure well and are now ready to be discharged from the hospital',
                                        r'Please follow the recommendations below to ensure a speedy and uneventful recovery',
                                        r'Division of Vascular and Endovascular Surgery[^\.]{0,200}please note'],
    "caotic endarterectomy template sentences": [r'You tolerated the procedure well and are now ready to be discharged from the hospital',
                                                 r'You are doing well and are now ready to be discharged from the hospital',
                                                 r'Please follow the recommendations below to ensure a speedy and uneventful recovery'],
    "neck surgery template sentences": [r'Rest is important and will help you feel better\. Walking is also important\. It will help prevent problems'],
    "TAVR template sentences": [r'If you stop these medications or miss[^\.]{0,30}, you risk causing a blood clot forming on your new valve',
                                r'These medications help to prevent blood clots from forming on the new valve'],
    "appendicitis template sentences": [r' preparing for discharge home with the following instructions'],
    "bowel obstruction template sentences": [r'You may return home to finish your recovery\. Please monitor'
                                             r'may or may not have had a bowel movement prior to[^\.]{0,20} discharge which is acceptable[^\.]{0,5} however it is important that[^\.]{0,30} have a bowel movement in'],
    "small bowel obstruction template sentences": [r'You have tolerated a regular diet, are passing gas [^\.]{0,30} (?:not taking any pain medications|pain is controlled with pain medications by mouth)\.'],

    # These are the most general regexes, motivated by the fact that usually after a list no meaningful fluent text follows
    # Had to split because look before not compatible with variable length
    "general headings": [r'^\w' + re_heading_general + re_item_element,
                         r'(?<=\. )' + re_heading_general + re_item_element],
    # When at least two items with the same delimiter cut from thereon
    "at least two items": [r'^(?:' + item + r'(?:[^\n]+\n){1,2}\n?){2,}' for item in ITEMIZE_ELEMENTS],
}

def create_delimiter_regex(delimiter_list):
    # Be careful with wildcards because include newlines (so no dotall flag)
    return re.compile('|'.join(delimiter_list), re.IGNORECASE|re.MULTILINE)
RE_SUFFIXES_DICT = {delimiter_name: create_delimiter_regex(delimiter_list) for delimiter_name, delimiter_list in SUFFIXES_DICT.items()}
