HMMs are widely used in Speech Recognition tasks. For example, in Isolated Word Recognition, we assume a vocabulary of size V, and we model each word in the vocabulary by a district HMM. We assume that we have K occurrences of each spoken word as training. 

The Speech Recognition system is made of the following steps:
- Extract features (typically MFCCs) for each training sample 
- Train an HMM for each of the words in the vocabulary
- For each sample in test, extract features and estimate likelihood to belong to each HMM

We then attribute the sample to the HMM with the highest likelihood. Note that we consider HMM observations as being generated from a Gaussian Mixture Model at each step.