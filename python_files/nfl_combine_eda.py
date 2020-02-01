#!/usr/bin/env python
# coding: utf-8

# # NFL Combine Classification Modeling
# 
# ## Exploratory Data Analysis

# ## Project Goals
# 
# - Determine the influence the NFL Combine has on a prospect getting drafted or not.
# - Determine the influence the NFL Combine has in terms of how early or how late a prospect gets drafted.
# - Discover which NFL Combine drills have the most impact on a prospect's draft status.

# ## Summary of Data
# 
# The dataset that was analyzed for this study contains 10,228 observations of NFL Combine data, dating from 1987-2018.

# ### Library Import

# In[ ]:


#Import libraries
# %run ../python_files/


# ## Data Import and Data Examination

# In[ ]:


# import data
#  = pd.read_csv('../data/.csv')

# quick overview of the dataset
# df


# 

# In[ ]:


# quick review of the variables in the dataset
# df.info()


# 

# In[ ]:


# quick review of the characteristics of our current continuous variables in the dataset
# df.describe()


# 

# In[ ]:


# check the number of NaN values in the dataset
# df.isna().sum()


# 

# ## Data Cleaning, Data Transformations, and Data Exploration
# 
# 

# In[ ]:


# Create dummy values for the categorical variables

# auto_df['mstatus'] = auto_df['mstatus'].map({'Yes': 1, 'No': 0})
# auto_df['sex'] = auto_df['sex'].map({'M': 1, 'F': 0})
# auto_df['education'] = auto_df['education'].map({'<High School': 0, 'High School': 0, 'Bachelors': 1, 'Masters': 1, 'PhD': 1})
# auto_df['job'] = auto_df['job'].map({'Student': 1, 'Blue Collar': 0, 'Clerical': 0, 'Doctor': 0, 'Home Maker': 0, 'Lawyer': 0, 'Manager': 0, 'Professional': 0})


# 

# In[ ]:


# Log Transformations for non-normalized variables. Then, drop the original variable from the dataset.

# def log_col(df, col):
#     '''Convert column to log values and
#     drop the original column
#     '''
#     df[f'{col}_log'] = np.log(df[col])
#     df.drop(col, axis=1, inplace=True)

# log_col(auto_df, 'tif')


# In[ ]:


# quick review of the characteristics of all variables in the dataset, 
# including the new dummy variables and log-transformed variables
# df.describe()


# 

# In[ ]:


# Correlations between all variables in auto_df dataset
# df.corr(method = 'pearson')


# In[ ]:


#Correlation Heatmap of all variables in auto_df dataset

# mask = np.zeros_like(auto_df.corr())
# triangle_indices = np.triu_indices_from(mask)
# mask[triangle_indices] = True

# plt.figure(figsize=(35,30))
# ax = sns.heatmap(auto_df.corr(method='pearson'), cmap="coolwarm", mask=mask, annot=True, annot_kws={"size": 18}, square=True, linewidths=4)
# sns.set_style('white')
# plt.xticks(fontsize=14, rotation=45)
# plt.yticks(fontsize=14, rotation=0)
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.show()


# ## Initial Train and Test Dataset Creation
# 
# 

# In[ ]:


#Split auto_insurance_df into train and test datasets for our logistic and linear regression models

#train and test datasets for logistic regression model
# crash = auto_df['crash']
# features_log = auto_df.drop(['crash', 'crash_cost'], axis = 1)
# x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(features_log, crash, test_size = 0.2, random_state = 10)


# ## Feature Selection
# 
# For modeling purposes, we used recursive feature elimination for both our logistic regression model and our simple linear regression model. This process uses cross-validation techniques, using accuracy as a metric, to eliminate variables that may hurt our model performance. Those variables get dropped from the dataset prior to modeling.

# ### Recursive Feature Elimination for Logistic Regression Model

# In[ ]:


# logreg_model = LogisticRegression()
# rfecv_log = RFECV(estimator=logreg_model, step=1, cv=StratifiedKFold(10), scoring='accuracy')
# rfecv_log.fit(x_train_log, y_train_log)


# In[ ]:


# feature_importance_log = list(zip(features_log, rfecv_log.support_))
# new_features_log = []
# for key,value in enumerate(feature_importance_log):
#     if(value[1]) == True:
#         new_features_log.append(value[0])
        
# print(new_features_log)


# In[ ]:


# linreg_model = LinearRegression()
# rfecv_lin = RFECV(estimator=linreg_model, step=1, min_features_to_select = 1, scoring='r2')
# rfecv_lin.fit(x_train_lin, y_train_lin)


# In[ ]:


# feature_importance_lin = list(zip(features_lin, rfecv_lin.support_))
# new_features_lin = []
# for key,value in enumerate(feature_importance_lin):
#     if(value[1]) == True:
#         new_features_lin.append(value[0])
        
# print(new_features_lin)


# ## Final Train and Test Datasets after Feature Selection
# 
# 

# In[ ]:


#final train and test datasets for logistic regression model
# x_train_log = x_train_log[new_features_log]
# x_test_log = x_test_log[new_features_log]

# print(x_train_log.shape)
# print(x_test_log.shape)

