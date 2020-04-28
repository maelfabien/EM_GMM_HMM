import python_speech_features as mfcc
from sklearn import preprocessing
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM

def get_MFCC(sr, audio):
    """
    Extracts the MFCC audio features from a file
    """
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 2, appendEnergy = False)
    features = preprocessing.scale(features)
    return features

def pipeline(n_components=4, max_iters=50):

	male_train = np.load('gender/male.npy')
	female_train = np.load('gender/female.npy')
	
	if max_iters == 0:
		return male_train, female_train, "", "", "", "", "", "", "", ""
	else:
		
		fs, data_male = read('gender/clips/male.wav')
		mfcc_male = get_MFCC(fs, data_male)

		gmm_male = GMM(n_components = n_components, max_iter = max_iters, covariance_type = 'diag', n_init = 3)
		gmm_male.fit(male_train)

		fs, data_female = read('gender/clips/female.wav')
		mfcc_female = get_MFCC(fs, data_female)

		gmm_female = GMM(n_components = n_components, max_iter = max_iters, covariance_type = 'diag', n_init = 3)
		gmm_female.fit(female_train)

		male_male = np.array(gmm_male.score(mfcc_male)).sum()
		female_male = np.array(gmm_female.score(mfcc_male)).sum()

		female_female = np.array(gmm_female.score(mfcc_female)).sum()
		male_female = np.array(gmm_male.score(mfcc_female)).sum()

		return male_train, female_train, male_male, male_female, female_male, female_female, gmm_male.means_, gmm_female.means_, gmm_male.covariances_, gmm_female.covariances_

def compute():
	if male_male >= male_female and female_female >= female_male:
	    return ["Male", "Female"]
	elif male_female >= male_male and female_female >= female_male:
		return ["Female", "Female"]
	elif male_female >= male_male and female_male >= female_female:
		return ["Female", "Male"]
	else:
		return ["Male", "Male"]