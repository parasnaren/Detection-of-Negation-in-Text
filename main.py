import argparse
import numpy as np
import os
from training_script import *
from predicting_script import *
from data_processing import *
from training_scope_script import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train')
    parser.add_argument('-s', '--test')
    #parser.add_argument('-d', '--directory', help="absolute path to corenlp directory. needs to be provided in raw mode", type=str, nargs='?')
    args = parser.parse_args()
    
    corenlp = input("Enter the absolute corenlp path: ")
    
    trainfile = args.train
    testfile = args.test
    
    print("Training file: ",trainfile)
    print("Testing file: ",testfile)
    
    #trainfile = 'training.txt'
    #testfile = 'cardboard.txt'
    filename = 'training.txt'
    
    #-t training.txt -s test.txt -c 'E:\MACHINE LEARNING\Phillips Hackathon\Round_3\stanford-corenlp-full-2018-10-05'
    #print(trainfile, testfile, corenlp)
    #corenlp = 'E:\MACHINE LEARNING\Phillips Hackathon\Round_3\stanford-corenlp-full-2018-10-05'
    
    # Train values
    print('\nTraining start.')
    sentence_dicts, cue_ssvm, cue_vec, cue_dict, affix_cue_dict = cue_trainer(trainfile,corenlp)
    scope_ssvm, scope_vec = scope_trainer(sentence_dicts)
    print('Training end.')
    
    # Predict values
    #testfile = 'SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'
    
    print('\nPredicting start.')
    cue_file = predict_test_cues(testfile, corenlp, cue_ssvm, cue_vec, cue_dict, affix_cue_dict)
    filename = predict_test_scope(scope_ssvm, scope_vec, cue_file)
    print('Predicting end.')
    
    finalfilename = append_cues_to_test(filename, testfile)
    print('Predicted file name : ', finalfilename)
    
    name = trainfile.split('.')[0] + '_actual.txt'
    os.remove(name)