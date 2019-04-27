import subprocess
import re

def file_to_sentence_dict(filename):
    with open(filename, 'r') as infile:
        sentence = {}
        cues = []
        mw_cues = []
        scopes = {}
        events = {}
        line_counter = 0
        counter = 0
        cue_counter = 0
        prev_cue_column = -1
        lower_limit = 35
        upper_limit = 7
        cue_offset = upper_limit - 5
        instances = []

        for line in infile:
            token_dict = {}
            tokens = line.split()
            #check for sentence end
            if len(tokens) == 0:
                for key in sentence:
                    head_index = int(sentence[key]['head']) - 1
                    if head_index > -1:
                        sentence[key]['head-pos'] = sentence[head_index][5]
                    else:
                        sentence[key]['head-pos'] = sentence[key][5]

                if(len(scopes) != len(cues)):
                    for i in range(len(cues)):
                        if not i in scopes:
                            scopes[i] = []

                sentence['cues'] = cues
                sentence['mw_cues'] = mw_cues
                sentence['scopes'] = scopes
                sentence['events'] = events
                    
                if len(cues) > 0:
                    sentence['neg'] = True
                else:
                    sentence['neg'] = False
                instances.append(sentence)
                sentence = {}
                counter = 0
                cue_counter = 0
                prev_cue_column = -1
                cues = []
                mw_cues = []
                scopes = {}
                events = {}
                line_counter += 1
                continue

            for i in range(len(tokens)):            
                if tokens[i] != "_" and  i < lower_limit:
                    token_dict[i+2] = tokens[i]
                elif tokens[i] != "***" and tokens[i] != "_" and i > upper_limit and (i-cue_offset) % 3 == 0:
                    if i == prev_cue_column:
                        cues[-1][2] = 'm'
                        prev_cue_column = i
                        if cues[-1][2] == 'm':
                            mw_cues.append([cues[-1][0],cues[-1][1]])
                        mw_cues.append([tokens[i], counter])
                    elif tokens[i] != tokens[1]:
                        cues.append([tokens[i], counter, 'a'])
                        prev_cue_column = i
                    else:
                        cues.append([tokens[i], counter, 's'])
                        prev_cue_column = i
                elif tokens[i] != "***" and tokens[i] != "_" and i > upper_limit and (i-cue_offset-1) % 3 == 0:
                    cue_counter = int((i-upper_limit+2)/3 - 1)
                    if cue_counter in scopes:
                        scopes[cue_counter].append([tokens[i], counter])
                    else:
                        scopes[cue_counter] = [[tokens[i], counter]]
                elif tokens[i] != "***" and tokens[i] != "_" and i > upper_limit and (i-cue_offset-2) % 3 == 0:
                    cue_counter = (i-upper_limit+3)/3
                    events[cue_counter] = tokens[i]
            token_dict[5] = tokens[4]
            token_dict['head'] = tokens[6]
            token_dict['deprel'] = tokens[7]
            sentence[counter] = token_dict
            counter += 1
            line_counter += 1
        return instances

def append_cues_to_test(tempfilename, testfile):
    finalfilename = testfile.split('.')[0] + '_pred.final'
    outfile = open(finalfilename, 'w')
    with open(testfile, 'r') as file1, open(tempfilename, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            temp = list()
            words1 = line1.split()
            words2 = line2.split()
            temp.extend(words1[0:7])
            temp.extend(words2[8:])
            sent = '\t'.join(temp)
            outfile.write(sent+'\n')
    outfile.close()
    return finalfilename

def adding_values_back(conllfilename, filename):
    whole = list()
    with open(filename, 'r') as orgfile:
        for line in orgfile:
            if line == '\n':
                whole.append([])
                continue
            words = line.split()
            whole.append(words)
        
    text = list()
    with open(conllfilename, 'r') as orgfile:
        for line in orgfile:
            if line == '\n':
                text.append([])
                continue
            words = line.split()
            if words[1].lower() == '-lsb-':
                words[1] = "["
            elif words[1].lower() == '-lrb-':
                words[1] = "("
            elif words[1].lower() == '-rsb-':
                words[1] = "]"
            elif words[1].lower() == '-rrb-':
                words[1] = ")"
            text.append(words)
        
    name = filename.split('.')[0]
    newfilename = name+'_actual.txt'
    actual = list()
    i, j = [0,0]
    while i < len(whole) and j < len(text):
        if(whole[i] == []):
            actual.append([])
        else:
            tmp = list()
            tmp.extend(whole[i][2:5])
            tmp.append('_')
            tmp.append(whole[i][5])
            tmp.append('_')
            tmp.extend(text[j][5:7])
            #tmp.extend(whole[i][7:])
            sent = '\t'.join(tmp)
            actual.append(sent)
        j+=1
        i+=1
    write_text_to_file(actual, newfilename)
    return newfilename


def convert_file_to_raw(filename):
    out_filename = filename.split('.')[0] + '_raw.txt'
    outfile = open(out_filename, 'w')
    with open(filename, 'r') as infile:
        for line in infile:
            if line == '\n':
                outfile.write('\n')
                continue
            words = line.split()
            outfile.write(words[3]+' ')
    return out_filename

          
def write_text_to_file(text, filename):
    with open(filename, 'a') as outfile:
        for row in text:
            if row == []:
                outfile.write('\n')
            else:
                outfile.write(row+'\n')
        
def adding_head_index(conllfilename, filename):
    whole = list()
    with open(filename, 'r') as orgfile:
        for line in orgfile:
            if line == '\n':
                whole.append([])
                continue
            words = line.split()
            whole.append(words)
        
    text = list()
    with open(conllfilename, 'r') as orgfile:
        for line in orgfile:
            if line == '\n':
                text.append([])
                continue
            words = line.split()
            if words[1].lower() == '-lsb-':
                words[1] = "["
            elif words[1].lower() == '-lrb-':
                words[1] = "("
            elif words[1].lower() == '-rsb-':
                words[1] = "]"
            elif words[1].lower() == '-rrb-':
                words[1] = ")"
            text.append(words)
        
    name = filename.split('.')[0]
    newfilename = name+'_actual.txt'
    actual = list()
    i, j = [0,0]
    while i < len(whole) and j < len(text):
        if(whole[i] == []):
            actual.append([])
        else:
            tmp = list()
            tmp.extend(whole[i][2:5])
            tmp.append('_')
            tmp.append(whole[i][5])
            tmp.append('_')
            tmp.extend(text[j][5:7])
            tmp.extend(whole[i][7:])
            sent = '\t'.join(tmp)
            actual.append(sent)
        j+=1
        i+=1
    write_text_to_file(actual, newfilename)
    return newfilename

#filename = "training.txt"
#corenlp = "E:\MACHINE LEARNING\Phillips Hackathon\Round_3\stanford-corenlp-full-2018-10-05"

def process_data(filename, corenlp):
    raw_filename = convert_file_to_raw(filename) 
    absolute_path = corenlp + "/*"
    args = ['java', '-cp', absolute_path, '-Xmx1800m', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',  '-annotators', 'tokenize,ssplit,pos,depparse', '-file', raw_filename, '-outputFormat', 'conll', '-tokenize.whitespace','-ssplit.eolonly']
    pipe = subprocess.call(args)
    conllfilename = raw_filename+'.conll'
    finalfilename = adding_head_index(conllfilename, filename)
    return finalfilename

def process_test_data(filename, corenlp):
    raw_filename = convert_file_to_raw(filename) 
    absolute_path = corenlp + "/*"
    args = ['java', '-cp', absolute_path, '-Xmx1800m', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',  '-annotators', 'tokenize,ssplit,pos,depparse', '-file', raw_filename, '-outputFormat', 'conll', '-tokenize.whitespace','-ssplit.eolonly']
    pipe = subprocess.call(args)
    conllfilename = raw_filename+'.conll'
    finalfilename = adding_values_back(conllfilename, filename)
    return finalfilename