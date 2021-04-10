# Dennis Orellana
## Data 670 Capstone Project Part 4

## Purpose: This is part 4 of the capstone project. 
# This script is data exploration after mbuilding models. 

# %%
# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import klib

# %%
#set work directory

os.chdir('C:\Documents\Data 670\Capstone Dataset')

# %%
# load the dataset from working directory 

job_posting= pd.read_csv('job_posting_supercleaned.csv')

# %%
"""## Exploratory Data Analysis ##"""

# %%
# Show columns
job_posting.columns

# %% 
# Top five job title
job_posting['job_title'].value_counts().head(5)

# %%
# Select the fraudulent rows
fraud = job_posting[job_posting.fraudulent == 1]
# %%
# print fraud
fraud
# %%
# Top five  fraudulent job title
fraud['job_title'].value_counts().head(5)

# %%
# Select Correlation Columns
fraud_cor = fraud[['job_title','location','job_id','telecommuting','benefits','I.T']]

# %%
# Plot the Fraudulent Job posting
klib.cat_plot(fraud_cor, top=5)


# %%
# Rescoures 
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
# https://www.youtube.com/watch?v=2AFGPdNn4FM
# https://klib.readthedocs.io/en/latest/
