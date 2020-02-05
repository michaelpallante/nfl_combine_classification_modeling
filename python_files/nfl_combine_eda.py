#!/usr/bin/env python
# coding: utf-8

# # NFL Combine Classification Modeling
# 
# ## Exploratory Data Analysis

# ## Project Goals
# 
# - Determine the influence the NFL Combine has on a lineman (offensive linemen and defensive linemen) prospect's draft status (getting drafted or not).
# - Discover which NFL Combine drills have the most impact on a lineman (offensive linemen and defensive linemen) prospect's draft position.
# - Reveal how much the NFL Combine factors in on a lineman (offensive linemen and defensive linemen) prospect's draft value (how early or how late a prospect gets drafted, if at all). **(*FUTURE WORK*)**

# ## Summary of Data
# 
# The dataset that was analyzed for this study contains 9,544 observations of NFL Combine and NFL Draft data, dating from 1987-2017. The NFL Combine data primarily displays the performance of players over that time period in combine drills. The NFL Draft data contains the draft pick information of players from that time span, including what round they were selected in and the team that picked them.

# ### Library Import
# 
# Below, we import the necessary libraries that will be used for our technical work.

# In[31]:


#Import libraries
# get_ipython().run_line_magic('run', '../python_files/libraries')
# get_ipython().run_line_magic('matplotlib', 'inline')
from libraries import *    #for use within .py file


# ## Data Importing, Data Merging, and Data Cleaning
# 
# Below, we import the NFL Combine and NFL Draft data, getting them both prepared and cleaned for modeling purposes.

# In[2]:


# import NFL Combine and NFL Draft data
nfl_combine_df = pd.read_csv('../data/nfl_combine_cleaned.csv')
nfl_draft_df = pd.read_csv('../data/nfl_draft_data.csv')

# quick overview of the NFL Combine dataset
# nfl_combine_df


# In the NFL Combine dataframe, we see players' performance and measurements from the combine, dating from 1987-2017.

# In[3]:


# quick overview of the NFL Draft dataset
# nfl_draft_df


# In the NFL Draft dataframe, we see players' draft pick information, dating from 1987-2017, including what round they were selected in and which team picked them.

# In[4]:


# merge the NFL Combine and NFL Draft datasets with an outer join so that all values and rows are retained
nfl_merged_df = pd.merge(nfl_combine_df, nfl_draft_df, how = 'left', on = ['player_name', 'last_name',
                                                                                'first_name', 'combine_year'])

# impute round and pick values to reflect undrafted players. round '13' means undrafted (draft lasts only 12 rounds)
# and pick '337' means undrafted (only 336 players get drafted). 'udfa' means no team drafted the players, therefore
# making them undrafted free agents (udfa).
nfl_merged_df['round'].fillna('13', inplace=True)
nfl_merged_df['pick'].fillna('337', inplace=True)
nfl_merged_df['team'].fillna('udfa', inplace=True)

# change round and pick variables to integer values
nfl_merged_df['round'] = nfl_merged_df['round'].astype(int)
nfl_merged_df['pick'] = nfl_merged_df['pick'].astype(int)

# quick overview of nfl_df dataset
# nfl_merged_df


# We merged the NFL Combine and NFL Draft dataframes into one merged NFL dataset. We instantly notice that some players do not have a draft round/pick or a team that selected them. This is because those players went undrafted in their respective draft years. To compensate for that, we need to impute these empty cells with the following values: round '13' means undrafted (draft lasts only 12 rounds); pick '337' means undrafted (only 336 players get drafted); 'udfa' means no team drafted the players, therefore making them undrafted free agents (udfa).

# In[5]:


# quick review of the variables in the NFL merged dataset
# nfl_merged_df.info()


# In[6]:


# quick review of the characteristics of the feature variables in the dataset
# nfl_merged_df.describe()


# In[7]:


# check the number of missing values in the NFL merged dataset
# nfl_merged_df.isna().sum()


# Since the 60_yard_shuttle column is missing almost 70% of the total observations, we decided that it would be best to remove this column, as majority of the players within our data did not participate in this drill. This also means we need to remove the 60_yard_shuttle_missed column, as it is no longer needed. We are keeping all other columns for further evaluation. One could argue that we should consider removing the 3_cone_drill column as well, due to a large number of missing values, but we have identified this drill as one that could be significant in determining player draft stock.

# In[8]:


nfl_merged_df = nfl_merged_df.drop(['60_yard_shuttle', '60_yard_shuttle_missed'], axis = 1)
# nfl_merged_df


# Our merged NFL dataframe containing both NFL Combine and NFL Draft data is now ready for further preparation.

# ## Data Transformations
# 
# Below, we use data transformations to create our two primary response variables for modeling purposes. First, we create the draft_status column, which shows whether a player was drafted or undrafted in the NFL Draft '1' means the player was drafted and '0' means the player was undrafted). We then create the draft_value column, which shows us how early or late a player was selected in the NFL Draft, and also if they went undrafted ['1' means the player was a Day 1 draft pick (selected in Round 1); '2' means the player was a Day 2 draft pick (selected in Rounds 2-3); '3' means the player was a Day 3 draft pick (selected in Rounds 4-7); '4' means the player was selected in Rounds 8-12 (the draft used to follow a 12 round format until 1993. 1993 draft was 8 rounds. 1994-present drafts are 7 rounds); '5' means the player was undrafted].

# In[9]:


# created draft_status column. '1' means the player was drafted and '0' means the player was undrafted
nfl_merged_df['draft_status'] = nfl_merged_df['round'].astype(str).map({'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                                                                        '7': 1, '8': 1, '9': 1, '10': 1, '11': 1,
                                                                        '12': 1, '13': 0})

# created draft_value column. '1' means the player was a Day 1 draft pick (selected in Round 1), '2' means the player
# was a Day 2 draft pick (selected in Rounds 2-3), '3' means the player was a Day 3 draft pick (selected in
# Rounds 4-7), '4' means the player was selected in Rounds 8-12 (the draft used to follow a 12 round format until
# 1993. 1993 draft was 8 rounds. 1994-present drafts are 7 rounds.), '5' means the player was undrafted.
nfl_merged_df['draft_value'] = nfl_merged_df['round'].astype(str).map({'1': 1, '2': 2, '3': 2, '4': 3, '5': 3, '6': 3, 
                                                                       '7': 3, '8': 4, '9': 4, '10': 4, '11': 4,
                                                                       '12': 4, '13': 5})


# In[10]:


# nfl_merged_df


# In[11]:


# nfl_merged_df.info()


# In[12]:


# nfl_merged_df.describe()


# Our updated NFL merged dataset is now ready for some intial evaluation, before we split the data into training and test datasets for modeling and also consider data imputation.

# ## Create Linemen (Offensive Linemen and Defensive Linemen) Dataframe
# 
# For the purposes our this study, we split our data into a dataframe containing only offensive and defensive linemen. This dataframe will be called 'linemen'.

# In[13]:


linemen = nfl_merged_df.loc[(nfl_merged_df.position == 'dl') |(nfl_merged_df.position == 'ol')]
linemen = linemen.reset_index(drop=True)
# linemen.head()


# In[14]:


# linemen.shape


# In[15]:


# linemen.position.value_counts()


# Our linemen dataframe contains 3098 total linemen, including 1642 offensive linemen and 1456 defensive linemen. This dataframe is now ready for further evaluation.

# ## Initial Findings and Data Visualizations

# ### Correlations

# In[16]:


# Correlations between all variables in the linemen dataset
# linemen.corr(method = 'pearson')


# In[17]:


#Correlation Heatmap of all variables in the linemen dataset

# mask = np.zeros_like(linemen.corr())
# triangle_indices = np.triu_indices_from(mask)
# mask[triangle_indices] = True

# plt.figure(figsize=(35,30))
# ax = sns.heatmap(linemen.corr(method='pearson'), cmap="coolwarm", mask=mask, annot=True, annot_kws={"size": 18}, square=True, linewidths=4)
# sns.set_style('white')
# plt.xticks(fontsize=14, rotation=45)
# plt.yticks(fontsize=14, rotation=0)
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.show()


# ### Distribution of Linemen Combine Drills

# In[ ]:


#


# ### Distribution of Linemen Draft Status

# In[58]:


#


# ### Distribution of Linemen Draft Status over Time

# In[59]:


#


# ## Train and Test Dataset Creation
# 
# We split our NFL linemen dataframe using a 80%-20% training and test split. We also randomized the selection of observations so that we did not bias the data. We completed this process twice: once for our draft_status model and once for our draft_value model.

# In[18]:


# Split nfl_merged_df into train and test datasets for our draft_status and draft_value models,
# using a randomized 80/20 split

#features for all models
features_linemen = linemen.iloc[:,6:24]

#train and test datasets for draft_status model
draft_status = linemen['draft_status']
x_train_ds, x_test_ds, y_train_ds, y_test_ds = train_test_split(features_linemen, draft_status, test_size = 0.2, random_state = 10)
x_train_ds = x_train_ds.reset_index(drop=True)
x_test_ds = x_test_ds.reset_index(drop=True)
y_train_ds = y_train_ds.reset_index(drop=True)
y_test_ds = y_test_ds.reset_index(drop=True)

#train and test datasets for draft_value model
draft_value = linemen['draft_value']
x_train_dv, x_test_dv, y_train_dv, y_test_dv = train_test_split(features_linemen, draft_value, test_size = 0.2, random_state = 10)
x_train_dv = x_train_dv.reset_index(drop=True)
x_test_dv = x_test_dv.reset_index(drop=True)
y_train_dv = y_train_dv.reset_index(drop=True)
y_test_dv = y_test_dv.reset_index(drop=True)


# In[19]:


# features_linemen.shape


# We have 2 different response variables that will each be used in their own model: one is draft_status and one is draft_value. There are 18 feature variables that will be used in our models as explanatory variables.

# ## Data Imputation
# 
# ### Linear Regression Imputation
# 
# We determined that linear regression imputation would be best for filling out missing observations for our models, as this gives us predicted values for our linemen in their combine drills, while still preserving the variance and randomization of our data over time.

# In[20]:


linemen = linemen.drop(['round','pick','team','draft_value'], axis=1)

drills = ['hand_size_inches', 'arm_length_inches', '40_yard_dash', 'bench_press_reps', 'vertical_leap_inches',
          'broad_jump_inches', '3_cone_drill', '20_yard_shuttle']


# In[21]:


# create function to display number of missing drills by position

def missing_drills(df,pos):
    print(pos)
    print(df.loc[df.position == pos, drills].shape, '\n')
    print(df.loc[df.position == pos, drills].isnull().sum(), '\n', '\n')
    
# compute number of missing drills for offensive and defensive linemen    
# missing_drills(linemen,'ol')
# missing_drills(linemen,'dl')


# In[22]:


# KNN imputation variable
imputer = KNNImputer(n_neighbors=5, copy=True)

# base imputation for our missing values in our draft_status model, using KNN imputation
knn_train_ds = pd.DataFrame(imputer.fit_transform(x_train_ds))
knn_test_ds = pd.DataFrame(imputer.fit_transform(x_test_ds))

# base imputation for our missing values in our draft_value model, using KNN imputation
knn_train_dv = pd.DataFrame(imputer.fit_transform(x_train_dv))
knn_test_dv = pd.DataFrame(imputer.fit_transform(x_test_dv))


# In[23]:


# linemen columns that contain missing data
missing_cols = linemen.isnull().sum()[linemen.isnull().sum() > 0].index
# missing_cols


# In[24]:


linemen = x_train_ds
linemen2 = x_test_ds


# In[25]:


for index, col in enumerate(x_train_ds.columns):
    if col in missing_cols:
#         print(index, col, 'MISSING')
        linemen[col + '_imp'] = knn_train_ds[index]
    else:
#         print(index,col)
        continue


# In[26]:


for index, col in enumerate(x_test_ds.columns):
    if col in missing_cols:
#         print(index, col, 'MISSING')
        linemen2[col + '_imp'] = knn_test_ds[index]
    else:
#         print(index,col)
        continue


# In[27]:


model_df = linemen.drop(missing_cols,axis=1)
model_df2 = linemen2.drop(missing_cols,axis=1)


# In[28]:


for feature in missing_cols:
        X = model_df.drop(feature+'_imp', axis=1)
        y = model_df[feature+'_imp']
        model = LinearRegression()
        model.fit(X,y)
        x_train_ds.loc[x_train_ds[feature].isnull(), feature] = model.predict(linemen[model_df.columns].drop(feature+'_imp', axis=1))[linemen[feature].isnull()]
        


# In[30]:


for feature in missing_cols:
        X = model_df2.drop(feature+'_imp', axis=1)
        y = model_df2[feature+'_imp']
        model = LinearRegression()
        model.fit(X,y)
        x_test_ds.loc[x_test_ds[feature].isnull(), feature] = model.predict(linemen2[model_df2.columns].drop(feature+'_imp', axis=1))[linemen2[feature].isnull()]

# Now, our 'x' training and test datasets for both of our models have been imputed.
#
# Below, we remove the KNN imputation columns that were used to predict our missing values in our original columns, as they are no longer needed.

x_train_ds = x_train_ds.iloc[:, :-8]
x_test_ds = x_test_ds.iloc[:, :-8]

# Now, our training and test datasets for both of our models are ready for modeling purposes. In order to prevent data leakage, we do not impute for our 'y' training and test datasets.
# 
# **To review our model implementation and model performance, please see our [Technical Notebook](https://github.com/michaelpallante/nfl_combine_classification_modeling/blob/master/notebooks/nfl_combine_technical_notebook.ipynb).**
