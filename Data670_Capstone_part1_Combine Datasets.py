# %%
# Dennis Orellana
## Capstone Project

## Purpose: This part 1 of the exploratory, data cleaning, and concatenate three datasets.
# %%
# Import Libraries
import os
import pandas as pd 
import numpy as np 
import seaborn as sb
import matplotlib.pyplot as plt
import klib

# %%
#set work directory
os.chdir('C:\Documents\Data 670\Capstone Dataset')

# %%
##load datasets from working directory 
indeed_data= pd.read_csv('Indeed Job Posting Dataset.csv')
fake_job_postings_data= pd.read_csv('fake_job_postings.csv')
data_scientist_jobs= pd.read_csv('data_scientist_united_states_job_postings_jobspikr.csv')

# %%
"""## **Data Exploration** """
# %%
# Convet Indeed dataset to Dataframe 
indeed_df = pd.DataFrame(indeed_data)

# %%
# Display the first five rows
indeed_df.head()

# %%
# Convert fake_job_postings dataset to Dataframe 
fake_jobs_df = pd.DataFrame(fake_job_postings_data)

# %%
# Display the first five rows
fake_jobs_df.head()

# %%
# Convet data_scientist_jobs dataset to Dataframe 
data_scientist_jobs_df= pd.DataFrame(data_scientist_jobs)

# %%
# Display the first five rows
data_scientist_jobs_df.head()

# %%
# Show Indeed columns

indeed_df.columns

# %%
# Show indeed data size 
indeed_df.shape

# %%
# Data types of Indeed
indeed_df.dtypes

# %%
# Show fake job columns
fake_jobs_df.columns

# %%
# Show fake job data size 
fake_jobs_df.shape

# %%
# Data types of fake job posting
fake_jobs_df.dtypes

# %%
# Show data_scientist_jobs columns
data_scientist_jobs_df.columns

# %%
# Show data_scientist_jobs data size
data_scientist_jobs_df.shape

# %%
# Data types of data_scientist_jobs 
data_scientist_jobs_df.dtypes

"""Missing Values Plot """
# %%
#Indeed missing values plot 
klib.missingval_plot(indeed_df)

# %%
# fake job positng missing values plot 
klib.missingval_plot(fake_jobs_df)

# %%
# data_scientist_jobs missing values plot 
klib.missingval_plot(data_scientist_jobs_df)

# %%
"""##Data Cleaning Indeed Data Cleaning"""
 # %%
# drop any cloumns with high ratio missing values
indeed_cleaned = klib.data_cleaning(indeed_df)

# %%
#remove features with a high ratio of missing values based on their informational content
indeed_deepcleaned = klib.mv_col_handling(indeed_cleaned)


# %%
# Drop Employer Column
indeed_supercleaned= indeed_deepcleaned.drop(columns=['employer_state'])

# %%
# Display columns types
indeed_supercleaned.dtypes

# %%
# Convert Columns into Category
indeed_supercleaned['job_title'] = indeed_supercleaned['job_title'].astype("category")
indeed_supercleaned['job_description'] = indeed_supercleaned['job_description'].astype("category")
indeed_supercleaned['location'] = indeed_supercleaned['location'].astype("category")
indeed_supercleaned['city'] = indeed_supercleaned['city'].astype("category")
indeed_supercleaned['zip_code'] = indeed_supercleaned['zip_code'].astype("category")
indeed_supercleaned['company_name'] = indeed_supercleaned['company_name'].astype("category")

# %%
# Clean the dataset
indeed_supercleaned = klib.data_cleaning(indeed_supercleaned)

# %%
# Display columns types after the converting 
indeed_supercleaned.dtypes

# %%
#plot missing values
klib.missingval_plot(indeed_supercleaned)

# %%
# Fill in missing values to Zip code with City columns
indeed_supercleaned['zip_code'].replace('NaN',np.nan,inplace=True)
indeed_supercleaned['zip_code'] = indeed_supercleaned.sort_values(by='city').groupby('city')['zip_code'].apply(lambda x : x.ffill().bfill())

# %%
# reduce the number of missing values
indeed_supercleaned['zip_code'].value_counts(dropna=False)


# %%
# Show any missing values
indeed_supercleaned.isnull().sum()

# %%
# add a new column as "job_board"
indeed_supercleaned['job_board'] = 'indeed'

# %%
# display the new indeed dataset
indeed_supercleaned.head()

# %%
"""## Fake Job Posting Data Cleaning """

# %%
# drop any cloumns with high ratio missing values
fake_jobscleaned = klib.data_cleaning(fake_jobs_df)

# %%
# Plot the missing values for each column
klib.missingval_plot(fake_jobscleaned)

# %%
# Remove any high ratio missing value columns
fake_jobdeepcleaned = klib.mv_col_handling(fake_jobscleaned)

# %%
# Plot the missing values for each column after remove high ratio
klib.missingval_plot(fake_jobdeepcleaned)

# %%
# Rename dataset
fake_job_supercleaned = fake_jobdeepcleaned


# %%
# Show any missing values
fake_job_supercleaned.isnull().sum()

# %%
"""## Data Scientist Job Data Cleaning """

# %%
# drop any cloumns with high ratio missing values
data_scientist_cleared = klib.data_cleaning(data_scientist_jobs_df)

# %%
# Plot the missing values for each column
klib.missingval_plot(data_scientist_cleared)

# %%
# Drop some columns
data_scientist_supercleared = data_scientist_cleared.drop(columns=['html_job_description','inferred_state', 'inferred_city', 'geo'])

# %%
# Plot the missing values for each column
klib.missingval_plot(data_scientist_supercleared)

# %%
# show missing values
data_scientist_supercleared.isnull().sum()

# %% 
"""## Review Data Size after Data Cleaning """

# %%
# Show data_scientist_supercleaned dataframe
data_scientist_supercleared.shape

# %%
# Show indeed_supercleaned dataframe
indeed_supercleaned.shape
 
# %%
# show  fake_job_supercleaned dataframe
fake_job_supercleaned.shape

# %%
"""##Rename some common columns between the three datasets## """

# %%
data_scientist_supercleared.columns

# %%
fake_job_supercleaned.columns

# %%
indeed_supercleaned.columns

# %%
# Rename Columns 
indeed_final = indeed_supercleaned.rename(columns={"Job Title": "job_title", "Job Description": "description", "Uniq Id" : "uniq_id", "Company Name" : "company_name"})
fake_job_final = fake_job_supercleaned.rename(columns={"title": "job_title", "description" : "job_description"})

# %%
# New funtion as data_scientist_final
data_scientist_final = data_scientist_supercleared

# %%
# Show rename columns 
indeed_final.columns

# %%
# Show rename columns 
fake_job_final.columns

# %%
# Show columns 
data_scientist_final.columns

# %%
"""## Concatenate three datasets into one dataset##"""

# %%
# concatenate all the datasets 
job_posting = pd.concat([indeed_final, fake_job_final, data_scientist_final], axis=0)

# %%
# Export job_posting as a csv file
job_posting.to_csv (r'C:\Documents\Data 670\Capstone Dataset\job_posting.csv', index = False, header=True)

# End of Script 