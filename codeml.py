import streamlit as st
import SessionState
import time

import pandas as pd
import numpy as np

#from sklearn.metrics import mean_squared_error,mean_squared_log_error
from math import sqrt

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


@st.cache
def loadmodel_10feats():

	df_test_10feats = pd.read_csv('testset_preprocessed_10feats.csv')
	df_Ytest_10feats = pd.read_csv('Y_test_10feats.csv')
	df_train_10feats = pd.read_csv('X_train_10feats.csv')
	df_Ytrain_10feats = pd.read_csv('Y_train_10feats.csv')
	X_test_10feats = pd.read_csv('X_test_10feats.csv')

	with open("Pickle_LR_Model_10feats.pkl", 'rb') as file:  
		Pickled_LR_Model_10feats = pickle.load(file)

	with open("Pickle_Preprocessor_10feats.pkl", 'rb') as file:  
		Pickled_Preprocessor_10feats = pickle.load(file)

	return(Pickled_LR_Model_10feats, Pickled_Preprocessor_10feats, df_test_10feats, df_Ytest_10feats, df_train_10feats, df_Ytrain_10feats, X_test_10feats)

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
	        _arr = p.transform(_arr).toarray()
	        b.append(_arr)

	return np.hstack(b).reshape((1, -1))

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
	session_state = SessionState.get(name="", predict1=False, predict2=False, buildmodel1 = False, buildmodel2 = False)

	st.write("## INAP Machine Learning Model Maker")
	left_column, right_column = st.beta_columns(2)
	with left_column:
		st.write("## House Prices Prediction")
	with right_column:
		image = Image.open('linreg.png')
		st.image(image, width = 120)
	st.write("### Dataset - Residential Homes in Ames, Iowa")

	buildmodel1 = st.button("Build model 7 feats")
	if buildmodel1:
		session_state.buildmodel1 = True

	if session_state.buildmodel1:

		my_bar = st.progress(0)

		for percent_complete in range(100):
			time.sleep(0.01)
			my_bar.progress(percent_complete + 1)



		lrmodel, prepro_model, all_data, df_test, df_Ytest, df_train, df_Ytrain, X_test = loadmodel()
	
		X_7feats = pd.concat([df_train, df_Ytrain], axis= 1)
		Xtest_7feats = pd.concat([df_test, df_Ytest], axis= 1)
	
		st.write(X_7feats)
		#st.write("Total number of observations: ", all_data.shape[0], "Selected number of features: ", all_data.shape[1])
		st.write("Trainig set - # of observations: ", X_7feats.shape[0], "# of features: ", X_7feats.shape[1])
		st.write("Testing set - # of observations: ", Xtest_7feats.shape[0], "# of features: ", Xtest_7feats.shape[1])
		pred, score_val = predict(lrmodel, df_test, df_Ytest)

		#st.write(score_val)
		#rmse = sqrt(mean_squared_error(df_Ytest, pred))
		#st.write(rmse)

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


		#predict1 = st.button("Predict!")

		if st.button("Predict!"):
			session_state.predict1 = True

		if session_state.predict1:

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


			if st.button("Built model 10 features"):

				session_state.predict2 = True

				my_bar = st.progress(0)

				for percent_complete in range(100):
					time.sleep(0.01)
					my_bar.progress(percent_complete + 1)

			if session_state.predict2:

				Pickled_LR_Model_10feats, Pickled_Preprocessor_10feats, df_test_10feats, df_Ytest_10feats, df_train_10feats, df_Ytrain_10feats, X_test_10feats = loadmodel_10feats()

				X_10feats = pd.concat([df_train_10feats, df_Ytrain_10feats], axis= 1)
				Xtest_10feats = pd.concat([df_test_10feats, df_Ytest_10feats], axis= 1)

				st.write(X_10feats)
				#st.write("Number of observations: ", X_10feats.shape[0], "Number of features: ", X_10feats.shape[1])

				st.write("Trainig set - # of observations: ", X_10feats.shape[0], "# of features: ", X_10feats.shape[1])
				st.write("Testing set - # of observations: ", Xtest_10feats.shape[0], "# of features: ", X_10feats.shape[1])

				pred, score_val = predict(Pickled_LR_Model_10feats, df_test_10feats, df_Ytest_10feats)

				#st.write(score_val)
				#rmse = sqrt(mean_squared_error(df_Ytest_10feats, pred))
				#st.write(rmse)

				left_column, right_column = st.beta_columns(2)
				with left_column:

					externalquality = st.radio('External quality', ("Ex", "Gd", "TA", "Fa"), index = 2 )

					#fireplaces = st.slider("Fireplaces", min_value=min(X_test_10feats["Fireplaces"]),  max_value=max(X_test_10feats["Fireplaces"]))
					fireplaces = st.radio('Fireplaces', range(3))


				with right_column:
					neighborhood_code = {'Bloomington Heights': "Blmngtn", 'Bluestem': "Blueste", 'Briardale':"BrDale", 'Brookside': "BrkSide", 'Clear Creek':"ClearCr", 'College Creek':"CollgCr", \
						'Crawford':"Crawfor", 'Edwards':"Edwards", 'Gilbert':"Gilbert", 'Iowa DOT and Rail Road':"IDOTRR", 'Meadow Village':"MeadowV", 'Mitchell':"Mitchel", 'Northridge':"NoRidge", 'Northpark Villa':"NPkVill" ,\
						'Northridge Heights':"NridgHt", 'Northwest Ames':"NWAmes", 'Old Town':"OldTown",'South & West of Iowa State University':"SWISU", 'Sawyer':"Sawyer", 'Sawyer West':"SawyerW", 'Somerset':"Somerst", 'Stone Brook':"StoneBr", 'Timberland':"Timber", 'Veenker':"Veenker"
						}
					
					neighborhood = st.selectbox('Neighborhood?',(list(neighborhood_code.keys())), index = 18)
					st.write('You selected:', neighborhood)


				left_column, right_column = st.beta_columns(2)
				with left_column:

					new_obs_10feats = pd.DataFrame([
						[ovq, gla, gc, yb, tbSF, _1flSF, _2flSF, externalquality, fireplaces, neighborhood_code[neighborhood]]
					], columns=[
						'OverallQual',
						'GrLivArea',
						'GarageCars',
						'YearBuilt',
						'TotalBsmtSF',
						'1stFlrSF',
						'2ndFlrSF',
						'ExterQual',
						'Fireplaces',
						'Neighborhood'
					], index=[0])





					st.write("New observation: ", new_obs_10feats.T)
					new_obs_preprocessed = preprocess(Pickled_Preprocessor_10feats, new_obs_10feats)
					#st.write(new_obs_preprocessed)
					st.write("Predicted Sale Price: ", Pickled_LR_Model_10feats.predict(new_obs_preprocessed))


					diffs = df_test_10feats.values - new_obs_preprocessed
					argmin = (diffs ** 2).sum(axis=1).argmin()

				with right_column:

					st.write("Closest observation: ", X_test_10feats.loc[argmin:argmin].T)

					st.write("Sale Price: ", df_Ytest_10feats.loc[argmin])


				if st.button("Learning curves comparison"):
					image = Image.open('performance_comparisonfeats.png')
					st.image(image, caption='Learning curves comparison - 7 feats vs 10 feats')






main()








