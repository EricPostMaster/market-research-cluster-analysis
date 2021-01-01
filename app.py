import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd


class Analysis:
	def __init__(self, df):
		self.df = df

	def clean_data(self, df):
		"""
		Input: Original uploaded CSV file converted to dataframe
		
		Output: New dataframe without UID and Constant columns, saved as attribute self.df_fct
				Also creates self.cleaned attribute and sets to True so I can prevent other actions
				from being taken before the data is cleaned.
		"""
		df_fct = df.drop(['UID','Const'], axis=1)
		self.df_fct = df_fct
		self.cleaned = True


def main():
	st.title("File Uploader")

	st.subheader("Upload Your Dataset")
	data_file = st.file_uploader("Upload CSV",type=['csv'])

	if st.button("Process"):
		if data_file is not None:
			file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
			st.write(file_details)

			# Save uploaded CSV to dataframe
			df = pd.read_csv(data_file)

			# Create an instance of the Analysis object and call it obj (this is confusing and should be changed)
			obj = Analysis(df=df)

			# Use the clean_data method on the Analysis object's original dataframe and print on screen
			obj.clean_data(obj.df)
			st.dataframe(obj.df)
			
			# Print the new dataframe
			st.dataframe(obj.df_fct)

		else:
			st.write("Please upload a CSV file for processing")
	


	# if st.button("Show cleaned dataframe"):
	# 	st.dataframe(df_fct)
		# if data_cleaned == 1:
		# 	st.dataframe(df_fct)
		# else:
		# 	st.write("Please process uploaded data first")



if __name__ == '__main__':
	main()