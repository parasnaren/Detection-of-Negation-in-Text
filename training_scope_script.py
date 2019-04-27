import numpy as np
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM

def scope_trainer(sentence_dicts):
    scope_instances, scope_labels, sentence_splits = extract_features_scope(sentence_dicts, 'training')
    scope_vec = DictVectorizer()
    fvs = scope_vec.fit_transform(scope_instances).toarray()
    X_train, y_train = split_data(fvs, scope_labels, sentence_splits)
    scope_ssvm = FrankWolfeSSVM(model=ChainCRF(), C=0.20, max_iter=10)
    scope_ssvm.fit(X_train, y_train)
    return scope_ssvm, scope_vec

def make_discrete_distance(dist):
    if dist <= 3:
        return 'A'
    elif dist <= 7:
        return 'B'
    elif dist > 7:
        return 'C'

def make_dir_graph_for_sentence(sentence):
    graph = nx.DiGraph()
    for key, value in sentence.items():
        if isinstance(key, int):
            head_index = int(value['head']) - 1
            if head_index > -1:
                graph.add_edge(str(head_index), str(key))
    return graph

def make_bidir_graph_for_sentence(sentence):
    graph = nx.DiGraph()
    for key, value in sentence.items():
        if isinstance(key, int):
            head_index = int(value['head']) - 1
            if head_index > -1:
                """tmp1 = dict()
                tmp2 = dict()
                tmp1['dir'] = '/'
                tmp2['dir'] = '\\'"""
                #graph.add_edge(str(head_index), str(key), attr_dict = {'dir':'/'})
                graph.add_edge(str(head_index), str(key), dir = '/')
                graph.add_edge(str(key), str(head_index), dir = '\\')
                #graph.add_edge(str(key), str(head_index), attr_dict = {'dir':'\\'})
    return graph

def get_shortest_path(graph, sentence, cue_index, curr_index):
    cue_head = int(sentence[cue_index]['head']) - 1
    if cue_head < 0 or curr_index < 0:
        return 'null'
    try:
        path_list = nx.dijkstra_path(graph, str(cue_head), str(curr_index))
        return make_discrete_distance(len(path_list) - 1)
    except nx.NetworkXNoPath:
        return 'null'

def get_dep_graph_path(graph, sentence, cue_index, curr_index):
    if cue_index < 0 or curr_index < 0:
        return 'null'
    try:
        path_list = nx.dijkstra_path(graph, str(curr_index), str(cue_index))
        prev_node = str(curr_index)
        dep_path = ""
        for node in path_list[1:]:
            direction = graph[prev_node][node]['dir']
            dep_path += direction
            if direction == '/':
                dep_path += sentence[int(node)]['deprel']
            else:
                dep_path += sentence[int(prev_node)]['deprel']
            prev_node = node
        return dep_path
    except nx.NetworkXNoPath:
        return 'null'


def extract_features_scope(sentence_dicts, mode='training'):
    instances = []
    sentence_splits = []
    for sent in sentence_dicts:
        if not sent['neg']:
            continue
        graph = make_dir_graph_for_sentence(sent)   # create directed path between token and cue
        bidir_graph = make_bidir_graph_for_sentence(sent)   # creates dependency graph for each
        for cue_i, (cue, cue_position, cue_type) in enumerate(sent['cues']):
            seq_length = -1
            for key, value in sent.items():
                features = {}
                if isinstance(key, int):
                    features['token'] = value[3]
                    features['lemma'] = value[4]
                    features['pos'] = value[5]
                    features['dir-dep-dist'] = get_shortest_path(graph, sent, cue_position, key)
                    features['dep-graph-path'] = get_dep_graph_path(bidir_graph, sent, cue_position, key)

                    """dist = key-cue_position
                    nor_index = -1
                    for k,v in sent.items():
                        if isinstance(k,int):
                            if v[3].lower() == "nor":
                                nor_index = k
                                break
                        
                    if cue == "neither" and nor_index > -1 and abs(key-nor_index) < abs(dist):
                        dist = key-nor_index
                    #token is to the left of cue
                    if dist < 0:
                        if abs(dist) <= 9:
                            features['left-cue-dist'] = 'A'
                        else:
                            features['left-cue-dist'] = 'B'
                        features['right-cue-dist'] = 'null'
                    #token is to the right of cue
                    elif dist > 0:
                        if dist <= 15:
                            features['right-cue-dist'] = 'A'
                        else:
                            features['right-cue-dist'] = 'B'
                        features['left-cue-dist'] = 'null'
                    else:
                        features['left-cue-dist'] = '0'
                        features['right-cue-dist'] = '0'"""
                    features['cue-type'] = cue_type
                    features['cue-pos'] = sent[cue_position][5]

                    if key == 0:
                        features['prev-word'] = 'null'
                        features['prev-word-pos'] = 'null'
                    else:
                        features['prev-word'] = "%s_*" %sent[key-1][4]
                        features['prev-word-pos'] = "%s_*" %sent[key-1][5]
                    if not (key+1) in sent:
                        features['next-word'] = 'null'
                        features['next-word-pos'] = 'null'
                    else:
                        features['next-word'] = "*_%s" %sent[key+1][4]
                        features['next-word-pos'] = "*_%s" %sent[key+1][5]
                    instances.append(features)
                    if key > seq_length:
                        seq_length = key
            sentence_splits.append(seq_length)
    if mode == 'training':
        labels = extract_labels_scope(sentence_dicts, mode)
        return instances, labels, sentence_splits
    return instances, sentence_splits

def extract_labels_scope(sentence_dicts, config):
    #inside scope 0
    #outside scope 1
    #start of scope 2
    #cue 3
    labels = []
    for sent in sentence_dicts:
        if not sent['neg']:
            continue
        for cue_i, (cue, cue_position, cue_type) in enumerate(sent['cues']):
            prev_label = 1
            for key, value in sent.items():
                if isinstance(key, int):
                    scope = sent['scopes'][cue_i]
                    if any(key in s for s in scope):
                        if prev_label == 1:
                            labels.append(2)
                            prev_label = 2
                        else:
                            labels.append(0)
                            prev_label = 0
                    elif key == cue_position:
                        labels.append(3)
                        prev_label = 3
                    else:
                        labels.append(1)
                        prev_label = 1
    return labels

def split_data(X, y, splits):
    i = 0
    j = 0
    X_train = []
    y_train = []
    offset = splits[j] + 1
    while j < len(splits) and offset <= len(X):
        offset = splits[j] + 1
        X_train.append(np.asarray(X[i:(i + offset)]))
        y_train.append(np.asarray(y[i:(i + offset)]))
        i += offset
        j += 1
    return np.asarray(X_train), np.asarray(y_train)