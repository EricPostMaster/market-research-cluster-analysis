import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
from operator import itemgetter
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import base64
import os
import json
import pickle
import uuid
import re

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(204, 204, 204);
                color: rgb(38, 39, 48);
                padding: 0.35em 0.48em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

class Analysis:
	def __init__(self):
		pass

	def create_df(self, df):
		self.df = df

	def clean_data(self):
		"""
		Input: Original uploaded CSV file converted to dataframe
		
		Output: New dataframe without UID and Constant columns, saved as attribute self.df_fct
				Also creates self.cleaned attribute and sets to True so I can prevent other actions
				from being taken before the data is cleaned.
		"""
		df_fct = self.df.drop(['UID','Const'], axis=1)
		self.df_fct = df_fct
		self.variables_to_examine = len(df_fct.columns)
		self.cleaned = True
	
	def factor_analysis(self):

		# Create factor analysis object and perform factor analysis
		fa = FactorAnalyzer(n_factors=len(self.df_fct.columns), rotation=None)
		fa.fit(self.df_fct)

		# Check Eigenvalues
		ev, v = fa.get_eigenvalues()

		# Create FactorAnalyzer object
		rotation = 'varimax'
		n_factors = sum(i >= 1 for i in ev)
		fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)

		# Fit factor analysis model to variables
		fa.fit(self.df_fct)

		# The loadings are the coefficients that make up the linear combination of original variables to get the factors (v1 = l1x1 + l2X2)
		loadings = fa.loadings_
		# Create dataframe of eigenvalues of the covariance matrix
		data = {'factor'                    : range(1,n_factors+1),
				'eigenvalues'               : fa.get_eigenvalues()[0][0:n_factors],
				'common_factor_eigenvalues' : fa.get_eigenvalues()[1][0:n_factors],
				'variance'                  : fa.get_factor_variance()[0],
				'proportional_variance'     : fa.get_factor_variance()[1],
				'cumulative_variance'       : fa.get_factor_variance()[2]
			}

		cov_matrix_eigenvals = pd.DataFrame(data=data).set_index('factor')
		
		# Scores for the factor analysis, converted to dataframe
		scores = pd.DataFrame(fa.transform(self.df_fct))
		self.scores = scores

	def clustering(self):

		sw=[]

		for i in range(2,7):
			
			# Create clustering objects
			cls1 = KMeans(n_clusters=i, random_state=0)
			cls2 = KMedoids(n_clusters=i, random_state=0)
			cls3 = AgglomerativeClustering(n_clusters=i, affinity = 'euclidean', linkage ='ward') #if linkage is ward, affinity must be Euclidean
			cls_algs = [['kMeans', cls1], ['kMedoids', cls2], ['Hierarchical', cls3]]
			
			# Fit and score clustering solutions for i clusters with each clustering algorithm
			for cls in cls_algs:
				
				# Fit the model to the factor analysis scores
				cls[1].fit(self.scores)
				
				# List of assigned clusters
				clusters = cls[1].fit_predict(self.scores)
				
				# Silhouette scores for each solution
				silhouette_avg = silhouette_score(self.scores,clusters)
				
				# Store solution info [algorithm, number of clusters, avg silhouette score, cluster predictions]
				algorithm = cls[0]
				i_stats = [algorithm, i, silhouette_avg, clusters]
				sw.append(i_stats)
				
				# Add columns of cluster assignments to df_fct datafram
				self.df_fct[algorithm+'_'+'cluster'+'_'+str(i)] = clusters


		# Reorder cluster lists by descending silhouette scores.  Clusters in first element should be assigned to training data.
		sw = sorted(sw, key=itemgetter(2), reverse=True)

		# Add the labels to the training dataset (you can ignore the warning when the cell runs)
		self.df_fct['cluster'] = sw[0][3]

	#########################################################################################
	# These functions are not being used right now because I'm trying to figure out scope
	#########################################################################################

	def split(self):
		# These are the variable columns and the optimal cluster assignment
		data_of_interest = self.df_fct.iloc[:,np.r_[:self.variables_to_examine,-1]]

		# Split data into 75% training, 12.5% validation, 12.5% test
		train, valid = train_test_split(data_of_interest, test_size=0.25, random_state=123)

		valid, test = train_test_split(valid, test_size=0.5, random_state=123)

		# X is unlabeled training data, y is true training labels 
		X, y = train.loc[:, train.columns != 'cluster'], train['cluster']

		X_valid, y_valid = valid.loc[:, train.columns != 'cluster'], valid['cluster']

		X_test, y_test = test.loc[:, test.columns != 'cluster'], test['cluster']


	def feature_importance(self):
		importance = pd.DataFrame({'variable': list(range(1,37)),
									'rf': clf1.feature_importances_,
									'gbt': clf2.feature_importances_,
									'avg': (importance['rf']+importance['gbt'])/2},
								).set_index('variable')

		# View top 10 variables when RF and GBT models are averaged
		top_10_avg = importance.sort_values(by='avg', ascending=False)['avg'].head(10)

		self.top_features = top_10_avg

	#########################################################################################
	#########################################################################################


	def classification(self):

		# These are the variable columns and the optimal cluster assignment
		data_of_interest = self.df_fct.iloc[:,np.r_[:self.variables_to_examine,-1]]

		# Split data into 75% training, 12.5% validation, 12.5% test
		train, valid = train_test_split(data_of_interest, test_size=0.25, random_state=123)

		valid, test = train_test_split(valid, test_size=0.5, random_state=123)

		# X is unlabeled training data, y is true training labels 
		X, y = train.loc[:, train.columns != 'cluster'], train['cluster']

		X_valid, y_valid = valid.loc[:, train.columns != 'cluster'], valid['cluster']

		X_test, y_test = test.loc[:, test.columns != 'cluster'], test['cluster']

		# self.split()

		clf_scores = []

		clf1 = RandomForestClassifier(random_state=0)
		clf2 = GradientBoostingClassifier(random_state=0)
		clf3 = SVC(random_state=0)
		clf4 = KNeighborsClassifier()

		classifiers = [['rf', clf1], ['gbt', clf2], ['svc', clf3], ['knn', clf4]]

		for classifier in classifiers:
			
			# Fit classifier to training data
			classifier[1].fit(X,y)    
			
			# Store classifier-specific results [algorithm object, classifier name, scores]
			results = [classifier[1], classifier[0], classifier[1].score(X_valid,y_valid)]

			# Overall classifier results
			clf_scores.append(results)

		# Sort classifier accuracy in descending order
		clf_scores = sorted(clf_scores, key=itemgetter(1), reverse=True) 

		# Run the model on final test data
		test_data_accuracy = round(clf_scores[0][0].score(X_test,y_test),5)*100

		self.test_accuracy = test_data_accuracy

		# self.feature_importance()
		importance = pd.DataFrame({'variable': list(range(1,37)),
							'rf': clf1.feature_importances_,
							'gbt': clf2.feature_importances_,
							# 'avg': (importance['rf']+importance['gbt'])/2
							},
						).set_index('variable')
		
		importance['avg'] = (importance['rf']+importance['gbt'])/2

		# View top 10 variables when RF and GBT models are averaged
		top_10_avg = importance.sort_values(by='avg', ascending=False)['avg'].head(10)

		self.top_features = top_10_avg



#########################################################################################
# This is where the class ends and the Streamlit app front end begins
#########################################################################################

def main():

	st.title("File Uploader")

	st.subheader("Upload Your Dataset")
	data_file = st.file_uploader("Upload CSV",type=['csv'])

	if st.button("Process"):
		if data_file is not None:

			# Save uploaded CSV to dataframe
			df = pd.read_csv(data_file)

			# Reassign value of obj.df attribute to the new dataframe df
			obj.create_df(df)

			# Display uploaded dataframe
			st.write("This is the original dataframe")
			st.dataframe(obj.df)

		else:
			st.write("Please upload a CSV file for processing")

	# Remove the first two columns to create .df_fct attribute
	obj.clean_data()

	# Factor Analysis
	obj.factor_analysis()

	# Clustering
	obj.clustering()
	st.write("Dataframe with cluster assignments added")
	st.dataframe(obj.df_fct)
		
	# Input for user to choose the filename
	filename = st.text_input('Enter output filename and ext (e.g. my-dataframe.csv)', 'test-file.csv')
	
	# Download button currently displays an error until the dataframe is processed
	# Make sure to add the UID and Const columns back onto the dataframe before downloading it
	download_button_str = download_button(obj.df_fct, filename, 'Click here to download', pickle_it=False)
	st.markdown(download_button_str, unsafe_allow_html=True)

	obj.classification()

	st.write(obj.__dict__)

	st.write(obj.top_features)

if __name__ == '__main__':
	# Create an instance of the Analysis object
	obj = Analysis()
	main()