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
from matplotlib import pyplot as plt
import time

st.set_page_config(
	page_title="Market Research Cluster Creation Tool",
	page_icon=":bulb:",
	layout="centered"
)

@st.cache
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
                padding: 0.55em 0.68em;
                position: relative;
                text-decoration: none;
                border-radius: 8px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                background-color: rgb(184, 184, 184);
                color: rgb(20, 21, 25);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(154, 154, 154);
                color: rgb(20, 21, 25);
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


@st.cache
def cluster_solver(df):

	"""
	Cleans, analyzes, clusters, and classifies.

	This function has four primary components:
		1. Clean the uploaded data by removing the first two columns that are not needed for analysis.
		2. Factor analysis to remove noise from uploaded data
		3. Clustering on the scores of the factor analysis. Silhouette scores are used to evaluate cluster solutions.
		4. Classification models for training, validation, and test data.

	Generates a dataframe with all clustering solutions added and optimal solution in final column.

	Params:
	------
	df: The uploaded CSV file. Must begin with two columns (UID and Const) and have the rest consist of only variables for modeling.

	Returns:
	-------
    df_fct: The modified dataframe consisting of only variables for factor analysis
	variables_to_examine: The number of variables used in factor analysis
	importance: Dataframe of relative importance of variables according to classification models
	file_uploaded: Boolean flag variable indicating that a file has been uploaded
	df_all_clusters: Final dataframe including all original columns as well as all clustering solutions

	Example:
	--------
	cluster_solver(df)

	"""

	#########################################################################################
	# Clean Data
	#########################################################################################

	# Uploaded CSV file
	df = df

	# Dataframe consisting only of the factors needed for analysis
	df_fct = df.drop(['UID','Const'], axis=1)
	
	# Number of factors needed in factor analysis
	variables_to_examine = len(df_fct.columns)
	

	#########################################################################################
	# Factor Analysis
	#########################################################################################

	# Create factor analysis object and perform factor analysis
	fa = FactorAnalyzer(n_factors=len(df_fct.columns), rotation=None)
	fa.fit(df_fct)

	# Check Eigenvalues
	ev, v = fa.get_eigenvalues()

	# Create FactorAnalyzer object with optimal number of factors
	rotation = 'varimax'
	n_factors = sum(i >= 1 for i in ev)
	fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)

	# Fit factor analysis model to variables
	fa.fit(df_fct)

	# Scores for the factor analysis, converted to dataframe
	scores = pd.DataFrame(fa.transform(df_fct))
	scores = scores


	#########################################################################################
	# Clustering
	#########################################################################################

	sw=[]

	for i in range(2,7):
		
		# Create clustering objects
		cls1 = KMeans(n_clusters=i, random_state=0)
		cls2 = KMedoids(n_clusters=i, random_state=0)
		cls3 = AgglomerativeClustering(n_clusters=i, affinity = 'euclidean', linkage ='ward')
		cls_algs = [['kMeans', cls1], ['kMedoids', cls2], ['Hierarchical', cls3]]
		
		# Fit and score clustering solutions for i clusters with each clustering algorithm
		for cls in cls_algs:
			
			# Fit the model to the factor analysis scores
			cls[1].fit(scores)
			
			# List of assigned clusters
			clusters = cls[1].fit_predict(scores)
			
			# Silhouette scores for each solution
			silhouette_avg = silhouette_score(scores,clusters)
			
			# Store solution info [algorithm, number of clusters, avg silhouette score, cluster predictions]
			algorithm = cls[0]
			i_stats = [algorithm, i, silhouette_avg, clusters]
			sw.append(i_stats)
			
			# Add columns of cluster assignments to df_fct dataframe
			df_fct[algorithm+'_'+'cluster'+'_'+str(i)] = clusters


	# Reorder cluster lists by descending silhouette scores.  Clusters in first element should be assigned to training data.
	sw = sorted(sw, key=itemgetter(2), reverse=True)

	# Add the labels to the training dataset
	df_fct['cluster'] = sw[0][3]

	# Complete dataframe with all originally uploaded columns and all clustering solutions
	df_all_clusters = pd.concat([df, df_fct.iloc[:,variables_to_examine:]], axis=1)


	#########################################################################################
	# Classification
	#########################################################################################

	# These are the variable columns and the optimal cluster assignment
	data_of_interest = df_fct.iloc[:,np.r_[:variables_to_examine,-1]]

	# Split data into 75% training, 12.5% validation, 12.5% test
	train, valid = train_test_split(data_of_interest, test_size=0.25, random_state=123)

	valid, test = train_test_split(valid, test_size=0.5, random_state=123)

	# X is unlabeled training data, y is true training labels 
	X, y = train.loc[:, train.columns != 'cluster'], train['cluster']

	X_valid, y_valid = valid.loc[:, train.columns != 'cluster'], valid['cluster']

	X_test, y_test = test.loc[:, test.columns != 'cluster'], test['cluster']

	# Somewhere to store classification solutions
	clf_scores = []

	# Create classifier objects
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

	# feature_importance()
	importance = pd.DataFrame({'variable': list(range(1,37)),
						'rf': clf1.feature_importances_,
						'gbt': clf2.feature_importances_,
						},)
					# ).set_index('variable')
	
	# Average variable importance of rf and gbt models
	importance['avg'] = (importance['rf']+importance['gbt'])/2

	# Average importance of all variables scaled from 0 to 1 (better interpretability)
	importance['Relative Importance'] = np.interp(importance['avg'], (importance['avg'].min(), importance['avg'].max()), (0, 1))

	# View top 10 variables when RF and GBT models are averaged
	top_10_vars = importance.sort_values(by='Relative Importance', ascending=False)['Relative Importance'].head(10)

	# Sort importance dataframe in descending order by Relative Importance
	importance.sort_values(by='Relative Importance', ascending=False, inplace=True)

	# File upload state (not needed in current version, maybe in future)
	file_uploaded = True

	return [df_fct, variables_to_examine, importance, file_uploaded, df_all_clusters]


#########################################################################################
# Streamlit app begins
#########################################################################################

st.title("Market Research Cluster Creation Tool")

st.subheader("How it works:")
st.write("""Upload your research participant dataset, and this tool will compute
an optimal clustering solution and identify the top 10 most important variables.""")
data_file = st.file_uploader("Upload CSV",type=['csv'])

# Once a file is uploaded, everything starts
if data_file is not None:

	# Save uploaded CSV to dataframe
	df = pd.read_csv(data_file)

	# Reassign value of df attribute to the new dataframe df
	r = cluster_solver(df)
	
	# Progress bar animation
	my_bar = st.progress(0)

	for percent_complete in range(100):
		time.sleep(0.01)
		my_bar.progress(percent_complete + 1)
	
	# Display and download clustered data 
	st.header("Clustered Data")
	st.write("""Cluster assignments have been added to the original data.
	Click the button below to download.""")
	st.dataframe(r[4]) 

	st.write("To download the dataset, enter the desired filename and click below.")

	# The download filename and button are in these two columns
	col1, col2 = st.beta_columns([2,1])

	with col1:
		filename = st.text_input('Enter output filename and ext (e.g. my-dataframe.csv)', 'clustered-data.csv')

	with col2:
		st.write(" ")
		st.write(" ")
		st.write(" ")
		download_button_str = download_button(r[4], filename, 'Download Data', pickle_it=False)
		st.markdown(download_button_str, unsafe_allow_html=True)

	# Informational section at the bottom of the page
	st.header("Most Impactful Stimuli")
	st.write("These features are the most important in identifying the clusters:")

	# Slider determines number of variables shown in plot and dataframe
	display_vars = st.slider("How many variables would you like to see?", min_value=3, max_value=r[1], value=10)

	# This is the subsetted data from the slider that is used in the plots
	graph_data = r[2].sort_values(by='Relative Importance', ascending=False).head(display_vars)

	# The variable column has to be turned into a string list for a categorical bar plot to render correctly
	names = graph_data['variable'].astype(str).tolist()
	values = graph_data['Relative Importance'].tolist()

	# Categorical bar plot
	fig, ax = plt.subplots(figsize=(8,4))
	ax.bar(names, values)
	plt.xlabel("Variable")
	plt.ylabel("Relative Importance")
	plt.title(f'Top {display_vars} Stimuli')
	plt.tight_layout()
	st.pyplot(fig)

	# Dataframe that shows the same thing as the plot but in tabular form
	st.subheader(f'Top {display_vars} Stimuli Dataframe')
	st.write(r[2].sort_values(by='Relative Importance', ascending=False)[['variable','Relative Importance']].head(display_vars))