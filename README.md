# EM for HMM and GMM

This is a Dash web application dedicated to illustrations of EM for HMM and GMM. 

![images](images/app.png)

## Files

- `app.py` : the core app component in Dash
- `countours.py` : functions to generate GMMs contour plots
- `datagen.py` : a random GMM data generation function
- `description_gmm.md` : text that goes into the explanation of the GMMs
- `gmm.py` : a version of the GMM implemented by hand
- `Procfile` : a file containing the launch phrase for the app
- `requirements.txt` : the list of requirements for the file
- `streamlit` : a folder containing Streamlit web application
- `images` : external images used
- `gender` : a folder for the gender prediction algorithm
	- `female.npy` : 2 MFCC features extracted on female trainings (AudioSet)
	- `male.npy` : 2 MFCC features extracted on female trainings (AudioSet)
	- `clips` : folder containing 2 wav files from AudioSet (Male and Female)

## How to use it?

It is currently deployed with Heroku on [this link](https://hmmgmm.herokuapp.com/).

One can also install it on his local machine. 
- Clone the repo
- `pip install -r requirements.txt`
- `python app.py`
