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


def add_df_to_Analysis(df):
	obj.create_df(df)



def main():

	st.title("File Uploader")

	st.subheader("Upload Your Dataset")
	data_file = st.file_uploader("Upload CSV",type=['csv'])

	if st.button("Process"):
		if data_file is not None:

			# This was from the boilerplate code.  Don't need it.
			# file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
			# st.write(file_details)

			# Save uploaded CSV to dataframe
			df = pd.read_csv(data_file)

			# Reassign value of obj.df attribute to the new dataframe df
			obj.create_df(df)

			# Use the clean_data method on the Analysis object's original dataframe and print on screen
			obj.clean_data()
			st.write("This is the original dataframe")
			st.dataframe(obj.df)
			
			# Print the new dataframe
			st.write("This is the dataframe with only the variables of interest")
			st.dataframe(obj.df_fct)
			# st.write(obj.__dict__)  # Won't need this when we're done

			obj.factor_analysis()
			st.write("These are the factor analysis scores")
			st.dataframe(obj.scores)

			obj.clustering()
			st.write("Dataframe with cluster assignments added")
			st.dataframe(obj.df_fct)

		else:
			st.write("Please upload a CSV file for processing")
	
	# if st.button("Show scores"):
	# 	st.write(obj.__dict__)
	# 	obj.factor_analysis()
	# 	st.dataframe(obj.scores)

if __name__ == '__main__':
	# Create an instance of the Analysis object and call it obj (this is confusing and should be changed)
	obj = Analysis()
	main()