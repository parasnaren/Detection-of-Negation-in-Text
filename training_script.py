import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from pystruct.models import BinaryClf
from pystruct.learners import NSlackSSVM
from data_processing import *

def get_cue_dict(sentence_dicts):
    cue_dict = {}
    affix_cue_dict = {'prefixes': [], 'infixes': [], 'suffixes': []}
    for sent in sentence_dicts:
        for (cue, cue_pos, cue_type) in sent['cues']:
            if cue_type == 'a':
                cue_token = sent[cue_pos][3].lower()
                if cue_token.startswith(cue.lower()):
                    if cue.lower() not in affix_cue_dict['prefixes']:
                        affix_cue_dict['prefixes'].append(cue.lower())
                elif cue_token.endswith(cue.lower()):
                    if cue.lower() not in affix_cue_dict['suffixes']:
                        affix_cue_dict['suffixes'].append(cue.lower())
                else:
                    if cue.lower() not in affix_cue_dict['infixes']:
                        affix_cue_dict['infixes'].append(cue.lower())
            elif cue_type == 's':
                if not cue.lower() in cue_dict:
                    cue_dict[cue.lower()] = cue_type
    return cue_dict, affix_cue_dict

def get_affix_cue(cue, affix_cue_dict):
    for prefix in affix_cue_dict['prefixes']:
        if cue.lower().startswith(prefix):
            return prefix
    for suffix in affix_cue_dict['suffixes']:
        if cue.lower().endswith(suffix):
            return suffix
    for infix in affix_cue_dict['infixes']:
        if infix in cue.lower() and not (cue.lower().startswith(infix) or cue.lower().endswith(infix)):
            return infix
    return None

def extract_features_cue(sentence_dicts, cue_dict, affix_cue_dict, mode='training'):
    instances = []
    for sent in sentence_dicts:
        for key, value in sent.items():
            features = {}
            if isinstance(key, int):
                cue_token = value[3].lower()
                if cue_token not in cue_dict and get_affix_cue(cue_token, affix_cue_dict) == None:
                    sent[key]['not-pred-cue'] = True
                    continue
    
                features['token'] = value[3].lower()
                features['lemma'] = value[4].lower()
                features['pos'] = value[5]
    
                if key == 0:
                    features['prev-word'] = 'null'
                else:
                    features['prev-word'] = "%s_*" %sent[key-1][4].lower()
                if not (key+1) in sent:
                    features['next-word'] = 'null'
                else:
                    features['next-word'] = "*_%s" %sent[key+1][4].lower()
                        
                affix = get_affix_cue(value[3].lower(), affix_cue_dict)
                if affix == None:
                    empty = ['null','null']
                    features['first-5'], features['last-5'] = empty
                    features['first-4'], features['last-4'] = empty
                    features['first-3'], features['last-3'] = empty
                    features['first-2'], features['last-2'] = empty
                    features['first-1'], features['last-1'] = empty
                    features['affix'] = 'null'                
                else:
                    word = value[3].lower().replace(affix, "")
                    features['first-5'], features['last-5'] = word[0:5], word[(len(word)-5):]
                    features['first-4'], features['last-4'] = word[0:4], word[(len(word)-4):]
                    features['first-3'], features['last-3'] = word[0:3], word[(len(word)-3):]
                    features['first-2'], features['last-2'] = word[0:2], word[(len(word)-2):]
                    features['first-1'], features['last-1'] = word[0:1], word[(len(word)-1):]
                    features['affix'] = affix                                        
                instances.append(features)    
                
    if mode == 'training':
        labels = extract_labels_cue(sentence_dicts, cue_dict, affix_cue_dict)
        return sentence_dicts, instances, labels
    return sentence_dicts, instances

def extract_labels_cue(sentence_dicts, cue_dict, affix_cue_dict):
    labels = []
    for sent in sentence_dicts:
        for key, value in sent.items():
            if isinstance(key, int):
                cue_token = value[3].lower()
                if cue_token not in cue_dict and get_affix_cue(cue_token,affix_cue_dict)==None:
                    continue
                if any(cue_position == key for (cue, cue_position, cue_type) in sent['cues']) or any(mw_pos == key for (mw_cue, mw_pos) in sent['mw_cues']):
                    labels.append(1)
                else:
                    labels.append(-1)
    return np.asarray(labels)

#
newfilename = "training_actual.txt"
filename = "dev.txt"
#


def cue_trainer(filename, corenlp):
    newfilename = process_data(filename, corenlp)
    sentence_dicts = file_to_sentence_dict(newfilename)
    cue_dict, affix_cue_dict = get_cue_dict(sentence_dicts)
    sentence_dicts, cue_instances, cue_labels = extract_features_cue(sentence_dicts, cue_dict, affix_cue_dict, 'training')
    cue_vec = DictVectorizer()
    model = cue_vec.fit_transform(cue_instances).toarray()
    cue_ssvm = NSlackSSVM(BinaryClf(), C=0.2, batch_size=-1)
    #cue_ssvm = SVC(C = 0.2)
    cue_ssvm.fit(model, cue_labels)
    return sentence_dicts, cue_ssvm, cue_vec, cue_dict, affix_cue_dict

    
    """pickle.dump(cue_ssvm, open("cue_model_%s.pkl" %filename, "wb"))
    joblib.dump(vec, "cue_vectorizer_%s.pkl" %filename)
    pickle.dump(cue_dict, open("cue_dict_%s.pkl" %filename, "wb"))
    pickle.dump(affix_cue_dict, open("affix_cue_dict_%s.pkl" %filename, "wb"))"""