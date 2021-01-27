import streamlit as st
import SessionState

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from math import sqrt
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

import streamlit as st
import pickle


import streamlit as st
import SessionState 

from PIL import Image

@st.cache
def loadmodel():

	df_test = pd.read_csv('testset_preprocessed.csv')
	df_Ytest = pd.read_csv('Ytestset.csv')

	df_train = pd.read_csv('X_train.csv')
	df_Ytrain = pd.read_csv('Y_train.csv')

	X_test = pd.read_csv('X_test.csv')

	all_data = pd.read_csv('all_dataset.csv')

	with open("Pickle_LR_Model.pkl", 'rb') as file:  
		Pickled_LR_Model = pickle.load(file)

	with open("Pickle_Preprocessor.pkl", 'rb') as file:  
		Pickled_Preprocessor = pickle.load(file)

	return(Pickled_LR_Model, Pickled_Preprocessor, all_data, df_test, df_Ytest, df_train, df_Ytrain, X_test)

def predict(model, df_test, df_Ytest):
	pred = model.predict(df_test)
	score_val = model.score(df_test, df_Ytest)
	return(pred, score_val)


@st.cache
def preprocess(preprocessor, arr):

	b = []
	for feat, p in preprocessor.items():
	    if(arr[feat].dtype != 'O'):
	        b.append(p.transform(arr[[feat]]))
	    else:
	        _arr = np.array(arr[feat]).reshape(-1,1)
	        b.append(p.transform(_arr).toarray())
	return np.array(b).reshape((1, -1))

@st.cache
def depreprocess(preprocessor, arr):

	b = []
	for feat, p in preprocessor.items():
	    if(arr[feat].dtype != 'O'):
	        b.append(p.inverse_transform(arr[[feat]]))
	    else:
	        _arr = np.array(arr[feat]).reshape(-1,1)
	        b.append(p.inverse_transform(_arr).toarray())
	return np.array(b).reshape((1, -1))


def main():

	#deleteoutliers = False


	st.write("## INAP Machine Learning Model Maker")
	left_column, right_column = st.beta_columns(2)
	with left_column:
		st.write("## House Prices Prediction")
	with right_column:
		image = Image.open('linreg.png')
		st.image(image, width = 120)
	st.write("### Dataset - Residential Homes in Ames, Iowa")


	lrmodel, prepro_model, all_data, df_test, df_Ytest, df_train, df_Ytrain, X_test = loadmodel()
	
	X_7feats = pd.concat([df_train, df_Ytrain], axis= 1)
	
	st.write(X_7feats)
	st.write("Number of observations: ", X_7feats.shape[0], "Number of features: ", X_7feats.shape[1])
	pred, score_val = predict(lrmodel, df_test, df_Ytest)

	st.write(score_val)
	rmse = sqrt(mean_squared_error(df_Ytest, pred))
	st.write(rmse)

	if st.button("Learning curve"):
		image = Image.open('performance_7feats.png')
		st.image(image, caption='Learning curve - 7 features')

	left_column, right_column = st.beta_columns(2)
	with left_column:
		ovq = st.slider("Overall quality", min_value=min(X_test["OverallQual"]),  max_value=max(X_test["OverallQual"]), value=5)
		gla = st.slider("GrLivArea", min_value=min(X_test["GrLivArea"]),  max_value=max(X_test["GrLivArea"]), value=1000)
		yb = st.slider("Year built", min_value=min(X_test["YearBuilt"]),  max_value=max(X_test["YearBuilt"]), value=1980)
		gc = st.slider("Garage Cars", min_value=min(X_test["GarageCars"]),  max_value=max(X_test["GarageCars"]), value=2)

	with right_column:
		
		tbSF = st.slider("TotalBsmtSF", min_value=min(X_test["TotalBsmtSF"]),  max_value=max(X_test["TotalBsmtSF"]), value=700)
		_1flSF = st.slider("1stFlrSF", min_value=min(X_test["1stFlrSF"]),  max_value=max(X_test["1stFlrSF"]), value=800)
		_2flSF = st.slider("2ndFlrSF", min_value=min(X_test["2ndFlrSF"]),  max_value=max(X_test["2ndFlrSF"]), value=0)

	if st.button("Predict!"):

		left_column, right_column = st.beta_columns(2)
		with left_column:
			new_obs = pd.Series({
				'OverallQual': ovq,
				'GrLivArea': gla,
				'GarageCars': gc,
				'YearBuilt': yb, 
				'TotalBsmtSF': tbSF,
				'1stFlrSF': _1flSF,
				'2ndFlrSF': _2flSF}).to_frame().T

			st.write("New observation: ", new_obs.T)
			new_obs_preprocessed = preprocess(prepro_model, new_obs)
			#st.write(new_obs_preprocessed)
			st.write("Predicted Sale Price: ", lrmodel.predict(new_obs_preprocessed))

		with right_column:
			MAIN_FEATS = ['OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']

			diffs = df_test[MAIN_FEATS].values - new_obs_preprocessed
			argmin = (diffs ** 2).sum(axis=1).argmin()


			st.write("Closest observation: ", X_test.loc[argmin:argmin, MAIN_FEATS].T)

			st.write("Sale Price: ", df_Ytest.loc[argmin])

			# diffs_preprocessed = preprocess(prepro_model, smallest_diff)
			# st.write(diffs_preprocessed)

			# st.write(lrmodel.predict(new_obs_preprocessed))

			



main()








