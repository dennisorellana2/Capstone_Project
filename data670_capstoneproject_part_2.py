# %%

# Dennis Orellana
## Capstone Project Part 2

## Purpose: This part 2 of the Exploratory & Data Cleaning Job Posting Dataset.

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
# load the dataset from working directory 

job_posting = pd.read_csv('job_posting.csv')

# %%
"""## Basic Exploratory Job Posting Dataset##"""

# %%
# show the first five rows
job_posting.head()

# %%
# # show the five last rows
job_posting.tail()

# %%
# show Dataset columns
job_posting.columns

# %%
# show datset size
job_posting.shape

# %%
"""##  More Exploratory & Clean Job Posting Dataset##"""
# %%
# show the columns
job_posting.columns

# %%
# Show the describe
job_posting.describe()

# %% 
# total missing values
job_posting.isnull().sum().sum()
 # %%
# Drop some columns
job_postingcleaned = job_posting.drop(columns=['url', 'category', 'country',
       'inferred_country', 'post_date', 'cursor'])

# %%
# show the leaned job posting columns
job_postingcleaned.columns 
  

# %%
# Check if any duplicate rows
job_postingcleaned.duplicated().sum()

# %%
#drop the duplicate values
job_postingcleaned.drop_duplicates(inplace=True)
# %%
# Check any duplicates after removed
job_postingcleaned.duplicated().sum()

# %%
# Plot Job Posting Fraud 
ax = job_postingcleaned['fraudulent'].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color=('#2ca02c', '#d62728'))
ax.set_title('Job Posting Fraud (0 = normal, 1 = fraud)', size=20, pad=30)
ax.set_ylabel('Number of Job Posting', fontsize=14)

for i in ax.patches:
    ax.text(i.get_x() + 0.19, i.get_height() + 700, str(round(i.get_height(), 2)), fontsize=15)

  # %% 
  # show the information
job_postingcleaned.info()

# %%
# Plot the missing values 
klib.missingval_plot(job_postingcleaned) 

# %% 
# Missing values for each column 
job_postingcleaned.isnull().sum()


# %%
# Show columns 
job_postingcleaned.columns

# %%
# Show job posting shape
job_postingcleaned.shape
 
# %%
# Clean data using Klib Function
job_posting_supercleaned = klib.data_cleaning(job_postingcleaned)
# %%
# Memory usage
job_posting_supercleaned.info(memory_usage='deep')

# %%
# Correlation Plots

klib.corr_plot(job_posting_supercleaned, annot=False, figsize=(15,12))
klib.corr_plot(job_posting_supercleaned, split='pos', annot=False, figsize=(15,12))
klib.corr_plot(job_posting_supercleaned, split='neg', annot=False, figsize=(15,12))

# %%
# Categorical Plots 
klib.cat_plot(job_posting_supercleaned)

# %%
# Show missing values in each columns
job_posting_supercleaned.isnull().sum()

# %%
# display first five values for job_id 
job_posting_supercleaned['job_id'].head

# %%
# Set the numerical data
df_num = job_posting_supercleaned[['job_id','telecommuting','has_company_logo','has_questions','fraudulent']]
# %%
# Checking for Outliers in numerical data
plt.figure(figsize=[16,8])
sb.boxplot(data = df_num)
plt.show()

# There aren't outliers

# %%
"""##  Handle Missing Values in Job Posting Dataset##"""

# %%
# Show missing values in each columns
job_posting_supercleaned.isnull().sum()

# %%
# Most frequent values in Location column 
job_posting_supercleaned["location"].value_counts ()

# %%
# Convert Columns into Category
job_posting_supercleaned['job_title'] = job_posting_supercleaned['job_title'].astype("category")
job_posting_supercleaned['job_description'] = job_posting_supercleaned['job_description'].astype("category")
job_posting_supercleaned['function'] = job_posting_supercleaned['function'].astype("category")
job_posting_supercleaned['department'] = job_posting_supercleaned['department'].astype("category")
job_posting_supercleaned['benefits'] = job_posting_supercleaned['benefits'].astype("category")
job_posting_supercleaned['job_type'] = job_posting_supercleaned['job_type'].astype("category")


# %%
# show data types
job_posting_supercleaned.dtypes
 # %%
# Replace the missing values with the most frequent values for each column
job_posting_supercleaned['location'] = job_posting_supercleaned['location'].fillna(job_posting_supercleaned["location"].value_counts().index[0])
job_posting_supercleaned['city'] = job_posting_supercleaned['city'].fillna(job_posting_supercleaned["city"].value_counts().index[0])
job_posting_supercleaned['state'] = job_posting_supercleaned['state'].fillna(job_posting_supercleaned["state"].value_counts().index[0])
job_posting_supercleaned['zip_code'] = job_posting_supercleaned['zip_code'].fillna(job_posting_supercleaned["zip_code"].value_counts().index[0])
job_posting_supercleaned['apply_url'] = job_posting_supercleaned['apply_url'].fillna(job_posting_supercleaned["apply_url"].value_counts().index[0])
job_posting_supercleaned['company_name'] = job_posting_supercleaned['company_name'].fillna(job_posting_supercleaned["company_name"].value_counts().index[0])
job_posting_supercleaned['companydescription'] = job_posting_supercleaned['companydescription'].fillna(job_posting_supercleaned["companydescription"].value_counts().index[0])
job_posting_supercleaned['uniq_id'] = job_posting_supercleaned['uniq_id'].fillna(job_posting_supercleaned["uniq_id"].value_counts().index[0])
job_posting_supercleaned['crawl_timestamp'] = job_posting_supercleaned['crawl_timestamp'].fillna(job_posting_supercleaned["crawl_timestamp"].value_counts().index[0])
job_posting_supercleaned['job_board'] = job_posting_supercleaned['job_board'].fillna(job_posting_supercleaned["job_board"].value_counts().index[0])
job_posting_supercleaned['job_id'] = job_posting_supercleaned['job_id'].fillna(job_posting_supercleaned["job_id"].value_counts().index[0])
job_posting_supercleaned['requirements'] = job_posting_supercleaned['requirements'].fillna(job_posting_supercleaned["requirements"].value_counts().index[0])
job_posting_supercleaned['telecommuting'] = job_posting_supercleaned['telecommuting'].fillna(job_posting_supercleaned["telecommuting"].value_counts().index[0])
job_posting_supercleaned['has_company_logo'] = job_posting_supercleaned['has_company_logo'].fillna(job_posting_supercleaned["has_company_logo"].value_counts().index[0])
job_posting_supercleaned['has_questions'] = job_posting_supercleaned['has_questions'].fillna(job_posting_supercleaned["has_questions"].value_counts().index[0])
job_posting_supercleaned['function'] = job_posting_supercleaned['function'].fillna(job_posting_supercleaned["function"].value_counts().index[0])
job_posting_supercleaned['job_description'] = job_posting_supercleaned['job_description'].fillna(job_posting_supercleaned["job_description"].value_counts().index[0])
job_posting_supercleaned['department'] = job_posting_supercleaned['department'].fillna(job_posting_supercleaned["department"].value_counts().index[0])
job_posting_supercleaned['benefits'] = job_posting_supercleaned['benefits'].fillna(job_posting_supercleaned["benefits"].value_counts().index[0])
job_posting_supercleaned['job_type'] = job_posting_supercleaned['job_type'].fillna(job_posting_supercleaned["job_type"].value_counts().index[0])
job_posting_supercleaned['job_title'] = job_posting_supercleaned['job_title'].fillna(job_posting_supercleaned["job_title"].value_counts().index[0])

# %%
# Show  fraudulent column values
job_posting_supercleaned['fraudulent'].value_counts()
# %%
# fraudulent shape
fraud = job_posting_supercleaned[job_posting_supercleaned['fraudulent']== 1]
fraud.shape

# %%
# non_fraudulent shape
not_fraud = job_posting_supercleaned[job_posting_supercleaned['fraudulent']== 0]
not_fraud.shape

# %%
# Balance the fraudulent column
fraud = fraud.sample(17014, replace=True)
# %%
# Show the fraud and non_fraud shape
fraud.shape, not_fraud.shape

# %%
# Append Method in the fraudulent column
job_posting_supercleaned = fraud.append(not_fraud)
job_posting_supercleaned.reset_index()

 # %%
# Show Missing values in each columns after cleaning 
job_posting_supercleaned.isnull().sum()

# There are no missing values


# %%
# show job_posting_supercleaned values
job_posting_supercleaned['job_type'].values

# %%
# Add column about Full Time or not
job_posting_supercleaned['Full Time'] = np.where(job_posting_supercleaned['job_type'] != 'Full Time', 'no','yes')
job_posting_supercleaned.head()

# %%
# Add column about IT job or not
job_posting_supercleaned['I.T'] = np.where(job_posting_supercleaned['department'] != 'Information Technology', 'no','yes')
job_posting_supercleaned.head()

# %%
# Show values about department
job_posting_supercleaned['department'].values

# %%
# Add column about IT job or not
job_posting_supercleaned['I.T'] = np.where(job_posting_supercleaned['department'] != 'Information Technology', 'no','yes')
job_posting_supercleaned.head()

# %%
# Export job_posting_supercleaned as a csv file
job_posting_supercleaned.to_csv (r'C:\Documents\Data 670\Capstone Dataset\job_posting_supercleaned.csv', index = False, header=True)


# %%
"""##  Convert the Job Posting Categorical Columns to Numerical ##"""

# %%
# Import LabelEncoder to convert numerical data
from sklearn.preprocessing import LabelEncoder

# %%
# Dataset type
job_posting_supercleaned.dtypes

# %%
# data size 
job_posting_supercleaned.shape

# %%
# Convert the dataset into numerical data
le = LabelEncoder()
job_posting_supercleaned['job_title'] = le.fit_transform(job_posting_supercleaned['job_title'])
job_posting_supercleaned['job_description'] = le.fit_transform(job_posting_supercleaned['job_description'])
job_posting_supercleaned['location'] = le.fit_transform(job_posting_supercleaned['location'])
job_posting_supercleaned['city'] = le.fit_transform(job_posting_supercleaned['city'])
job_posting_supercleaned['state'] = le.fit_transform(job_posting_supercleaned['state'])
job_posting_supercleaned['zip_code'] = le.fit_transform(job_posting_supercleaned['zip_code'])
job_posting_supercleaned['apply_url'] = le.fit_transform(job_posting_supercleaned['apply_url'])
job_posting_supercleaned['company_name'] = le.fit_transform(job_posting_supercleaned['company_name'])
job_posting_supercleaned['companydescription'] = le.fit_transform(job_posting_supercleaned['companydescription'])
job_posting_supercleaned['uniq_id'] = le.fit_transform(job_posting_supercleaned['uniq_id'])
job_posting_supercleaned['crawl_timestamp'] = le.fit_transform(job_posting_supercleaned['crawl_timestamp'])
job_posting_supercleaned['job_board'] = le.fit_transform(job_posting_supercleaned['job_board'])
job_posting_supercleaned['job_id'] = le.fit_transform(job_posting_supercleaned['job_id'])
job_posting_supercleaned['requirements'] = le.fit_transform(job_posting_supercleaned['requirements'])
job_posting_supercleaned['telecommuting'] = le.fit_transform(job_posting_supercleaned['telecommuting'])
job_posting_supercleaned['has_company_logo'] = le.fit_transform(job_posting_supercleaned['has_company_logo'])
job_posting_supercleaned['has_questions'] = le.fit_transform(job_posting_supercleaned['has_questions'])
job_posting_supercleaned['function'] = le.fit_transform(job_posting_supercleaned['function'])
job_posting_supercleaned['fraudulent'] = le.fit_transform(job_posting_supercleaned['fraudulent'])
job_posting_supercleaned['department'] = le.fit_transform(job_posting_supercleaned['department'])
job_posting_supercleaned['benefits'] = le.fit_transform(job_posting_supercleaned['benefits'])
job_posting_supercleaned['job_type'] = le.fit_transform(job_posting_supercleaned['job_type'])
job_posting_supercleaned['Full Time'] = le.fit_transform(job_posting_supercleaned['Full Time'])
job_posting_supercleaned['I.T'] = le.fit_transform(job_posting_supercleaned['I.T'])


# %%
# rename the dataset
job_posting_final = job_posting_supercleaned

# %%
# Show the data types
job_posting_final.dtypes

# %% 
# Show the first 5 rows
job_posting_final.head()

# %%
# Show the tail 5 rows
job_posting_final.tail()

# %%
# show the dataset shape
job_posting_final.shape

# %%
"""## More Data Visualization ##"""

# %%
# distribution plot
klib.dist_plot(job_posting_final)


# %%
# Correlations with feature plot
# The target is the fraudulent column
klib.corr_plot(job_posting_final, target='fraudulent')

# %%
# Displaying the postive and negative correlations plots
klib.corr_plot(job_posting_final, split='pos') # displaying only positive correlations
klib.corr_plot(job_posting_final, split='neg') # displaying only negative correlations

# %%
# Export job_posting_final as a csv file
job_posting_final.to_csv (r'C:\Documents\Data 670\Capstone Dataset\job_posting_final.csv', index = False, header=True)

# End of Script 
