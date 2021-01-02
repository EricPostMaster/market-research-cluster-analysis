import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd
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

# def get_table_download_link(df):
# 	"""Generates a link allowing the data in a given panda dataframe to be downloaded
# 	in:  dataframe
# 	out: href string
# 	"""
# 	csv = df.to_csv(index=False)
# 	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
# 	href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
# 	st.markdown(href)

import base64
import os
import json
import pickle
import uuid
import re

import streamlit as st
import pandas as pd

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

		for i in range (2,7):
			i_stats = []
			algorithm = "kMedoids"
			kMedoids = KMedoids(n_clusters=i, random_state=0)
			kMedoids.fit(self.scores)
			clusters=kMedoids.fit_predict(self.scores)
			silhouette_avg = silhouette_score(self.scores,clusters)  # 1 is a perfect score, -1 is worst score
			i_stats.append(algorithm)
			i_stats.append(i)
			i_stats.append(silhouette_avg)
			i_stats.append(clusters)
			sw.append(i_stats)
			self.df_fct[algorithm+'_'+'cluster'+'_'+str(i)] = clusters
			print(f"{i} k-medoid clusters: {round(silhouette_avg,3)}")

		for i in range (2,7):
			i_stats = []
			algorithm = "kMeans"
			kMeans = KMeans(n_clusters=i, random_state=0)
			kMeans.fit(self.scores)
			clusters=kMeans.labels_
			silhouette_avg = silhouette_score(self.scores,clusters)  # 1 is a perfect score, -1 is worst score
			i_stats.append(algorithm)
			i_stats.append(i)
			i_stats.append(silhouette_avg)
			i_stats.append(clusters)
			sw.append(i_stats)
			self.df_fct[algorithm+'_'+'cluster'+'_'+str(i)] = clusters
			print(f"{i} k-means clusters: {round(silhouette_avg,3)}")
			
		for i in range (2,7):
			i_stats = []
			algorithm = "hierarchical"
			hc = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage ='ward') #if linkage is ward, affinity must be Euclidean
			hc.fit_predict(self.scores)
			clusters=hc.labels_
			silhouette_avg = silhouette_score(self.scores,clusters)  # 1 is a perfect score, -1 is worst score
			i_stats.append(algorithm)
			i_stats.append(i)
			i_stats.append(silhouette_avg)
			i_stats.append(clusters)
			sw.append(i_stats)
			self.df_fct[algorithm+'_'+'cluster'+'_'+str(i)] = clusters
			print(f"{i} hierarchical clusters: {round(silhouette_avg,3)}")

		# Reorder cluster lists by descending silhouette scores.  Clusters in first element should be assigned to training data.
		sw = sorted(sw, key=itemgetter(2), reverse=True)

		# Add the labels to the training dataset (you can ignore the warning when the cell runs)
		self.df_fct['cluster'] = sw[0][3]

	def classification(self):
		clf_scores = []

		# These are the variable columns and the optimal cluster assignment
		data_of_interest = df_fct.iloc[:,np.r_[:36,-1]]

		# Split data into 80% training, 20% test
		train, test = train_test_split(data_of_interest, test_size=0.2, random_state=123)

		# X is unlabeled training data, y is true training labels 
		X, y = train.loc[:, train.columns != 'cluster'], train['cluster']

		X_test, y_test = test.loc[:, test.columns != 'cluster'], test['cluster']



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
	download_button_str = download_button(obj.df_fct, filename, 'Click here to download', pickle_it=False)
	st.markdown(download_button_str, unsafe_allow_html=True)



if __name__ == '__main__':
	# Create an instance of the Analysis object
	obj = Analysis()
	main()