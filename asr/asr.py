from __future__ import print_function
import warnings
import os
import python_speech_features as mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
import pickle

def extract_mfcc(full_audio_path):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc.mfcc(wave, sample_rate, 0.025, 0.01, 20, appendEnergy = False)
    return mfcc_features

def build_data(dir):
    # Filter out the wav audio files under the dir
    fileList = [dir+f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        label = fileName.split("/")[2].split('_')[0]
        feature = extract_mfcc(fileName)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset

def build_data(dir):
    
    fileList = [dir+f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        label = fileName.split("/")[2].split('_')[0]
        feature = extract_mfcc(fileName)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset

def train_GMMHMM(dataset, GMM_mix_num = 6, n_iter = 15):
    GMMHMM_Models = {}
    states_num = 5
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)

    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=n_iter)

        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model

    return GMMHMM_Models

def test_accuracy(GMM_mix_num = 6, n_iter = 15):

    with open('asr/train.pickle', 'rb') as handle:
        trainDataSet = pickle.load(handle)

    with open('asr/test.pickle', 'rb') as handle:
        testDataSet = pickle.load(handle)
    
    hmmModels = train_GMMHMM(trainDataSet, GMM_mix_num, n_iter)

    with open('asr/hmm.pickle', 'wb') as handle:
        pickle.dump(hmmModels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    count = 0
    score_cnt = 0
    for label in testDataSet.keys():
        
        feature = testDataSet[label]
        for feat in feature:
            count += 1
            scoreList = {}
            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(feat)
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)
            
            if predict == label:
                score_cnt+=1

    return 100.0*score_cnt/count

def run_asr(file):

    with open('asr/train.pickle', 'rb') as handle:
        trainDataSet = pickle.load(handle)

    with open('asr/hmm.pickle', 'rb') as handle:
        hmmModels = pickle.load(handle)

    feature = extract_mfcc(file)
    score_cnt = 0

    scoreList = {}
    for model_label in hmmModels.keys():
        model = hmmModels[model_label]
        score = model.score(feature)
        scoreList[model_label] = score
    
    predict = max(scoreList, key=scoreList.get)
    return predict

if __name__ == '__main__':
    test_accuracy()