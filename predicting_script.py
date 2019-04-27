from data_processing import *
from training_script import *
from training_scope_script import *

def make_complete_labelarray(sentences, labels):
    y = []
    label_counter = 0
    for sent in sentences:
        sent_labels = []
        for key, value in sent.items():
            if isinstance(key, int):
                if 'not-pred-cue' in value:
                    sent_labels.append(-2)
                else:
                    if labels[label_counter] == -1:
                        sent_labels.append(-1)
                    else:
                        sent_labels.append(1)
                    label_counter += 1
        y.append(sent_labels)
    return y

def check_by_no_means(sentence, index):
    if index == 0:
        return False
    if sentence[index][3].lower() == "no" and sentence[index-1][3].lower() == "by" and sentence[index+1][3].lower() == "means":
        return True
    return False

def check_neither_nor(sentence, index):
    if sentence[index][3].lower() == "nor" and any(sentence[key][3].lower() == "neither" for key in sentence if isinstance(key,int)):
        return True
    return False

def find_neither_index(sentence):
    for key,value in sentence.items():
        if isinstance(key,int):
            if value[3].lower() == "neither":
                return key
    return -1

def count_multiword_cues(sentence, labels):
    mwc_counter = 0
    has_mwc = False
    for key,value in sentence.items():
        if isinstance(key,int):
            if check_by_no_means(sentence, key):
                labels[key-1] = 1
                labels[key] = 1
                labels[key+1] = 1
                mwc_counter += 1
                has_mwc = True
            if check_neither_nor(sentence, key):
                neither_i = find_neither_index(sentence)
                if not (labels[neither_i] == 1 and labels[key] == 1):
                    mwc_counter += 1
                has_mwc = True
                labels[neither_i] = 1
                labels[key] = 1

    return mwc_counter, has_mwc


def mwc_start(token, prev_token):
    mw_lexicon = ['neither', 'by', 'rather', 'on']    
    return any(token.lower() == w for w in mw_lexicon) or (prev_token == "by" and token == "no")


def convert_cues_to_fileformat(sentences, labels, affix_cue_lexicon, filename):
    infile = open(filename, "r")
    output_filename = filename.split("_")[0] + "_cues.txt"
    outfile = open(output_filename, "w")
    sent_counter = 0
    line_counter = 0
    upper_limit = 8
    
    n_cues = sum(i > 0 for i in labels[sent_counter])
    n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
    if has_mwc:
        n_cues += n_mwc - 1
    written_cues = n_cues*[False]
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            line_counter = 0
            if sent_counter < len(labels):
                n_cues = sum(i > 0 for i in labels[sent_counter])
                n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
                if has_mwc:
                    n_cues += n_mwc - 1
                written_cues = n_cues*[False]
            outfile.write("\n")
        else:
            written_cue_on_line = False
            for i in range(upper_limit):
                outfile.write("%s\t" %tokens[i])
            if n_cues == 0:
                outfile.write("***\n")
            else:
                for cue_i in range(n_cues):
                    if labels[sent_counter][line_counter] < 0:
                        outfile.write("_\t_\t_\t")
                    else:
                        if written_cues[cue_i] or written_cue_on_line:                        
                            outfile.write("_\t_\t_\t")
                        else:
                            affix = get_affix_cue(tokens[1].lower(), affix_cue_lexicon)
                            if affix != None:
                                outfile.write("%s\t_\t_\t" %affix)
                                written_cues[cue_i] = True
                            else:
                                outfile.write("%s\t_\t_\t" %tokens[1])
                                prev_token = sentences[sent_counter][line_counter-1][3].lower() if line_counter > 0 else 'null'
                                if not mwc_start(tokens[1].lower(), prev_token):
                                    written_cues[cue_i] = True
                            written_cue_on_line = True
                line_counter += 1
                outfile.write("\n")
    infile.close()
    outfile.close()
    return output_filename

def in_scope_token(token_label, cue_type):
    return token_label == 0 or token_label == 2 or (token_label == 3 and cue_type == 'a')

def convert_scopes_to_fileformat(sentences, labels, cue_file):
    output_filename = cue_file.split('_')[0] + ".final"
    infile = open(cue_file, "r")
    outfile = open(output_filename, "w")
    sent_counter = 0
    line_counter = 0
    scope_counter = 0
    upper_limit = 8
    n_cues = 0
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            scope_counter += n_cues
            line_counter = 0
            n_cues = 0
            outfile.write("\n")
        elif tokens[-1] == "***":
            outfile.write(line)
        else:
            sent = sentences[sent_counter]
            cues = sent['cues']
            n_cues = len(cues)
            for i in range(upper_limit):
                outfile.write("%s\t" %tokens[i])
            for cue_i in range(n_cues):
                outfile.write("%s\t" %tokens[upper_limit + 3*cue_i])
                if in_scope_token(labels[scope_counter][line_counter], cues[cue_i][2]):
                    if cues[cue_i][2] == 'a' and sent[int(cues[cue_i][1])][3] == tokens[1]:
                        outfile.write("%s\t" %(tokens[1].replace(cues[cue_i][0], "")))
                    elif tokens[upper_limit + 3*cue_i] != "_":
                        outfile.write("_\t")
                    else:
                        outfile.write("%s\t" %tokens[1])
                else:
                    outfile.write("_\t")

                outfile.write("%s\t" %tokens[upper_limit + 2 + 3*cue_i])
                scope_counter += 1
            
            scope_counter -= n_cues
            line_counter += 1
            outfile.write("\n")

    infile.close()
    outfile.close()
    return output_filename

def read_test_data(filename):
    with open(filename, 'r') as infile:
        sentence = {}
        ct = 0
        instances = []
        for line in infile:
            token_dict = {}
            words = line.split()
            if len(words) == 0:
                for key in sentence:
                    head_node = int(sentence[key]['head'])-1
                    if head_node > -1:
                        sentence[key]['head-pos'] = sentence[head_node][5]
                    else:
                        sentence[key]['head-pos'] = sentence[key][5]

                instances.append(sentence)
                sentence = {}
                ct = 0
                continue
            for i in range(3):
                if words[i] != "_":
                    token_dict[i+2] = words[i]
            token_dict[5] = words[4]
            token_dict['head'] = words[6]
            token_dict['deprel'] = words[7]
            sentence[ct] = token_dict
            ct += 1
    return instances
            
def read_cuepredicted_data(filename):
    lower_limit = 3 
    upper_limit = 7
    cue_offset = upper_limit-5
    with open(filename, 'r') as infile:
        sentence = {}
        cues = []
        mw_cues = []
        line_counter = 0
        counter = 0
        prev_cue_column = -1
        instances = []

        for line in infile:
            token_dict = {}
            words = line.split()
            if len(words) == 0:
                for key in sentence:
                    head_node = int(sentence[key]['head']) - 1
                    if head_node > -1:
                        sentence[key]['head-pos'] = sentence[head_node][5]
                    else:
                        sentence[key]['head-pos'] = sentence[key][5]
                sentence['cues'] = cues
                sentence['mw_cues'] = mw_cues
                if len(cues) > 0:
                    sentence['neg'] = True
                else:
                    sentence['neg'] = False

                instances.append(sentence)
                sentence = {}
                counter = 0
                prev_cue_column = -1
                cues = []
                mw_cues = []
                line_counter += 1
                continue

            for i in range(len(words)):            
                if words[i] != "_" and i < lower_limit:
                    #add an offset of 2 to make token dicts match the original CD dicts
                    token_dict[i+2] = words[i]
                #cue column
                elif words[i] != "***" and words[i] != "_" and i > upper_limit and (i-cue_offset) % 3 == 0:
                    if i == prev_cue_column:
                        #same column has another cue token. cue must be mw
                        cues[-1][2] = 'm'
                        prev_cue_column = i
                        mw_cues.append([cues[-1][0],cues[-1][1]])
                        mw_cues.append([words[i], counter])
                    elif words[i] != words[1]:
                        #cue does not match current token. must be affixal cue
                        cues.append([words[i], counter, 'a'])
                        prev_cue_column = i
                    else:
                        cues.append([words[i], counter, 's'])
                        prev_cue_column = i
                        
            token_dict[5] = words[4]
            token_dict['head'] = words[6]
            token_dict['deprel'] = words[7]

            sentence[counter] = token_dict
            counter += 1
            line_counter += 1
        return instances
    
    
def predict_test_cues(testfile, corenlp, cue_ssvm, cue_vec, cue_dict, affix_cue_dict):
    filename = process_test_data(testfile, corenlp)
    sentence_dicts = read_test_data(filename)
    sentence_dicts, instances = extract_features_cue(sentence_dicts, cue_dict, affix_cue_dict, 'prediction')
    fvs = cue_vec.transform(instances).toarray()
    y_pred = cue_ssvm.predict(fvs)
    test_y_pred = make_complete_labelarray(sentence_dicts, y_pred)
    cue_file = convert_cues_to_fileformat(sentence_dicts, test_y_pred, affix_cue_dict, filename)
    return cue_file
    
def predict_test_scope(scope_ssvm, scope_vec, cue_file):
    sentence_dicts = read_cuepredicted_data(cue_file)
    instances, test_splits = extract_features_scope(sentence_dicts, 'prediction')
    test_fvs = scope_vec.transform(instances).toarray()
    X_test, y_test = split_data(test_fvs, [], test_splits)
    y_pred = scope_ssvm.predict(X_test)
    filename = convert_scopes_to_fileformat(sentence_dicts, y_pred, cue_file)
    return filename