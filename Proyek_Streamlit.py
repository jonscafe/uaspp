#!/usr/bin/env python
# coding: utf-8

# # UAS Project Streamlit: 
# - **Nama:** [Jonathan Sebastian Marbun]
# - **Dataset:** [Dataset lagu-lagu dan peringkat lagu di Spotify (https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset, https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023, https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs]
# - **URL Website:** [Di isi jika web streamlit di upload]
# 
# 

# ## Menentukan Pertanyaan Bisnis

# - Bagaimana trend musik di dunia berdasarkan Spotify dari tahun 1960-2023?
# - Musik seperti apa yang populer dalam rentang waktu tersebut?

# ## Import Semua Packages/Library yang Digunakan

# In[1]:

import subprocess

# Assuming your requirements.txt is in the same directory as your script
subprocess.run(["pip", "install", "-r", "requirements.txt"])

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# folder read
import os
for dirname, _, filenames in os.walk('dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Data Wrangling

# ### Gathering Data

# In[2]:

st.title("Spotify Chart: Song from 60's til 2023")

hits_dataset_filenames=['dataset-of-10s','dataset-of-00s','dataset-of-90s','dataset-of-80s','dataset-of-70s','dataset-of-60s']
hits_dataset_dict={}
for name in hits_dataset_filenames:
    df=pd.read_csv(f'dataset/{name}.csv')
    df.drop(df.iloc[:, 14:18], inplace=True, axis=1)
    
    df.drop(['uri'], inplace=True, axis=1)
    df.drop(['loudness'], inplace=True, axis=1)
    
    print(df.shape)
    hits_dataset_dict[name]=df


# ### Assessing Data

# In[3]:


full_hits_dataset = pd.concat(hits_dataset_dict.values(), ignore_index=True, axis=0)
track_artist_fhd=full_hits_dataset.pop('track') +full_hits_dataset.pop('artist')
track_artist_fhd.to_csv('dataset/track_artist_fhd.csv')
full_hits_dataset.info()

top_songs_dataset=pd.read_csv('dataset/spotify-2023.csv',encoding='latin-1')
top_songs_views_dataset=top_songs_dataset.pop('streams')
print(top_songs_dataset.columns)
top_songs_dataset.drop(top_songs_dataset.iloc[:, 2:13], inplace=True, axis=1)
# top_songs_dataset.drop(['track_id','duration_ms'], inplace=True, axis=1)

# top_songs_dataset.columns


# In[4]:


most_songs_dataset=pd.read_csv('dataset/spotify_songs.csv')

most_songs_dataset.drop(most_songs_dataset.iloc[:, 3:11], inplace=True, axis=1)
most_songs_dataset.drop(['track_id','duration_ms','loudness'], inplace=True, axis=1)



most_songs_dataset


# In[5]:


for l in most_songs_dataset.columns:
    if l not in full_hits_dataset.columns:
        print(l)
        
most_songs_dataset.rename(columns={"track_name":"track","track_artist":"artist"}, inplace=True)
for l in most_songs_dataset.columns:
    if l not in full_hits_dataset.columns:
        print(l)
        
track_artist_msd=most_songs_dataset.pop('track') +" "+ most_songs_dataset.pop('artist') 
track_artist_msd.to_csv('dataset/track_artist_msd.csv')

# track_artist_msd
 


# In[6]:


features=['tempo','key','mode','danceability','valence','energy','acousticness','instrumentalness','liveness','speechiness']


# In[7]:


#no loudness in this
for l in top_songs_dataset.columns:
    if l not in full_hits_dataset.columns:
        print(l)
        
# top_songs_dataset.rename(columns={"track_name":"track","track_artist":"artist"}, inplace=True)
top_songs_dataset
top_songs_dataset=top_songs_dataset.set_axis(['track','artist','tempo','key','mode','danceability','valence','energy','acousticness','instrumentalness','liveness','speechiness'], axis="columns")
for l in top_songs_dataset.columns:
    if l not in full_hits_dataset.columns:
        print(l)
track_artist_tsd=top_songs_dataset.pop('track')+" " +top_songs_dataset.pop('artist')
track_artist_tsd.to_csv('dataset/track_artist_tsd.csv')
top_songs_views_dataset=pd.concat([track_artist_tsd,top_songs_views_dataset], axis=1, join="inner",ignore_index=True)
top_songs_views_dataset=top_songs_views_dataset.set_axis(['track - artist', 'views'],axis='columns')
top_songs_views_dataset.to_csv('dataset/top_songs_views_dataset.csv')

# top_songs_views_dataset     
# top_songs_views_dataset.columns


# ### Cleaning Data

# In[8]:


chords=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


# In[9]:


top_songs_dataset.loc[top_songs_dataset['mode'] == 'Major', 'mode'] = 1
top_songs_dataset.loc[top_songs_dataset['mode'] == 'Minor', 'mode'] = 0
top_songs_dataset['key'] = top_songs_dataset['key'].fillna(-1)
for chord in chords:
    top_songs_dataset.loc[top_songs_dataset['key'] == chord, 'key'] = chords.index(chord)
labs = list(top_songs_dataset.columns)

for lab in labs:
    print(top_songs_dataset[lab].unique())


# In[10]:


feat_dict={}
for lab in list(full_hits_dataset.columns):
    if lab in ['target','tempo','key','mode']: 
        feat_dict[lab]=1
        continue
    feat_dict[lab]=100
full_hits_dataset.mul(feat_dict)
# print(len(feat_dict))
feat_dict.pop('target')
print(feat_dict)
top_songs_dataset.mul(feat_dict)
most_songs_dataset.mul(feat_dict)
top_songs_dataset=top_songs_dataset[most_songs_dataset.columns]
# full_hits_dataset


# ## Exploratory Data Analysis (EDA)

# ### Explore ...

# In[11]:


from sklearn.model_selection import train_test_split

y=full_hits_dataset.pop('target')
# full_hits_dataset.drop(['mode','key'], inplace=True, axis=1)
y.to_csv('dataset/target.csv')
y.to_csv('dataset/target.csv')
full_hits_dataset.to_csv('dataset/full_hits_dataset.csv')

X_train, X_test, y_train, y_test = train_test_split(full_hits_dataset, y, test_size=0.2, random_state=42)


# ## Visualization & Explanatory Analysis

# ### Pertanyaan 1:

# In[17]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you already have the dataframes full_hits_dataset, top_songs_dataset, most_songs_dataset

# Convert the dataframes to 1D arrays for scatter plot
full_hits_flat = full_hits_dataset.to_numpy().flatten()
top_songs_flat = top_songs_dataset.to_numpy().flatten()
most_songs_flat = most_songs_dataset.to_numpy().flatten()

# Create scatter plots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Scatter plot for full_hits_dataset
axes[0].scatter(range(len(full_hits_flat)), full_hits_flat, alpha=0.5)
axes[0].set_title('full_hits_dataset after multiplication')

# Scatter plot for top_songs_dataset
axes[1].scatter(range(len(top_songs_flat)), top_songs_flat, alpha=0.5)
axes[1].set_title('top_songs_dataset after multiplication')

# Scatter plot for most_songs_dataset
axes[2].scatter(range(len(most_songs_flat)), most_songs_flat, alpha=0.5)
axes[2].set_title('most_songs_dataset after multiplication')

plt.tight_layout()

# Save the plot to a temporary file
temp_file_path = "temp_scatter_plot.png"
plt.savefig(temp_file_path, format="png")

# Display the plot in Streamlit
st.image(temp_file_path, use_column_width=True)

# Remove the temporary file after displaying
import os
os.remove(temp_file_path)



# ### Pertanyaan 2:

# In[18]:


def plot_feature_by_category(dataset, title):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=dataset.mean().reset_index(), x='index', y=0)
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)
    plt.show()

# Plot for full_hits_dataset
plot_feature_by_category(full_hits_dataset, 'Popularity of Music - Full Hits Dataset')

# Plot for top_songs_dataset
plot_feature_by_category(top_songs_dataset, 'Popularity of Music - Top Songs Dataset')

# Plot for most_songs_dataset
plot_feature_by_category(most_songs_dataset, 'Popularity of Music - Most Songs Dataset')


# ## Membuat Model 

# In[19]:


# GaussianProcessClassifier() too much memory can't run
# KNeighborsClassifier(),RadiusNeighborsClassifier() no time to try

models=[svm.SVC(),svm.NuSVC()]
for model in models:
    dec = model
    dec=dec.fit(X_train,y_train)
    y_pred=dec.predict(X_test)


    print(classification_report(y_test, y_pred))
    conf_mat=confusion_matrix(y_test, y_pred)
    print(conf_mat)
    print(conf_mat.ravel())


# In[20]:


# HistGradientBoostingClassifier(),RandomForestClassifier(),ExtraTreesClassifier(),GradientBoostingClassifier(),AdaBoostClassifier() best
# svm.LinearSVC() very good not consistent
#tree.DecisionTreeClassifier(),SGDClassifier() +


models=[HistGradientBoostingClassifier(),RandomForestClassifier(),ExtraTreesClassifier(),GradientBoostingClassifier(),AdaBoostClassifier(),svm.LinearSVC(),SGDClassifier(),tree.DecisionTreeClassifier()]
for model in models:
    dec = model
    dec=dec.fit(X_train,y_train)
    y_pred=dec.predict(X_test)


    print(classification_report(y_test, y_pred))
    conf_mat=confusion_matrix(y_test, y_pred)
    print(conf_mat)
    print(conf_mat.ravel())


# ### Training Model

# In[21]:


pred=[]
models=[HistGradientBoostingClassifier(),RandomForestClassifier()]
# models=[HistGradientBoostingClassifier()]
for model in models:
    dec = model
    dec=dec.fit(X_train,y_train)
    y_pred=dec.predict(most_songs_dataset)
    pred.append(y_pred)

# full_msd_hgbc=track_artist_msd+most_songs_dataset+y_pred[0]
# full_msd_rfc=track_artist_msd+most_songs_dataset+y_pred[1]


# In[24]:


pred[0]=pd.DataFrame(pred[0],columns=['hit'])
pred[1]=pd.DataFrame(pred[1],columns=['hit'])

preds=[]
models=[HistGradientBoostingClassifier(),RandomForestClassifier()]
# models=[HistGradientBoostingClassifier()]
for model in models:
    dec = model
    dec=dec.fit(X_train,y_train)
    y_pred=dec.predict(top_songs_dataset)
    preds.append(y_pred)


# In[25]:


preds[0]=pd.DataFrame(pred[0],columns=['hit'])
preds[1]=pd.DataFrame(pred[1],columns=['hit'])


# ### Evaluasi Model

# In[26]:


# result = pd.concat([df1, df4], axis=1, join="inner")

full_msd_hgbc=pd.concat([track_artist_msd,pred[0]], axis=1, join="inner",ignore_index=True)
full_msd_rfc=pd.concat([track_artist_msd,pred[1]], axis=1, join="inner",ignore_index=True)
# full_msd_hgbc.rename(columns={"0": "track - artist", "1": "Hit"},inplace=True)
# full_msd_rfc.rename(columns={"0": "track - artist", "1": "Hit"},inplace=True)
print(full_msd_hgbc.shape)
full_msd_hgbc=full_msd_hgbc.set_axis(['track - artist', 'hit'], axis='columns')
full_msd_rfc=full_msd_rfc.set_axis(['track - artist', 'hit'], axis='columns')
# full_msd_hgbc.compare(full_msd_rfc)
print(full_msd_hgbc['track - artist'].isin(top_songs_views_dataset['track - artist'].values).unique(),
full_msd_rfc['track - artist'].isin(top_songs_views_dataset['track - artist'].values).unique())


# In[27]:


# result = pd.concat([df1, df4], axis=1, join="inner")

full_tsd_hgbc=pd.concat([track_artist_tsd,preds[0]], axis=1, join="inner",ignore_index=True)
full_tsd_rfc=pd.concat([track_artist_tsd,preds[1]], axis=1, join="inner",ignore_index=True)
# full_msd_hgbc.rename(columns={"0": "track - artist", "1": "Hit"},inplace=True)
# full_msd_rfc.rename(columns={"0": "track - artist", "1": "Hit"},inplace=True)
print(full_tsd_hgbc.shape)
full_tsd_hgbc=full_tsd_hgbc.set_axis(['track - artist', 'hit'], axis='columns')
full_tsd_rfc=full_tsd_rfc.set_axis(['track - artist', 'hit'], axis='columns')
# full_msd_hgbc.compare(full_msd_rfc)
print(full_tsd_hgbc['track - artist'].isin(top_songs_views_dataset['track - artist'].values).unique(),
full_tsd_rfc['track - artist'].isin(top_songs_views_dataset['track - artist'].values).unique())


# ### Menyimpan Model

# In[29]:


full_msd_rfc.set_axis(['track - artist', 'hit'],axis='columns')
full_msd_hgbc.set_axis(['track - artist', 'hit'],axis='columns')
print(top_songs_views_dataset.shape,full_msd_hgbc.shape,full_msd_hgbc['hit'].value_counts(),full_msd_rfc.shape,full_msd_rfc['hit'].value_counts())
merged_msd_hgbc = full_msd_hgbc.merge(top_songs_views_dataset,how="inner")
merged_msd_rfc=full_msd_rfc.merge(top_songs_views_dataset,how="inner")
print(merged_msd_hgbc['hit'].value_counts(),merged_msd_hgbc.shape,
merged_msd_rfc['hit'].value_counts(),merged_msd_rfc.shape)

full_tsd_rfc.set_axis(['track - artist', 'hit'],axis='columns')
full_tsd_hgbc.set_axis(['track - artist', 'hit'],axis='columns')
print(top_songs_views_dataset.shape,full_tsd_hgbc.shape,full_tsd_hgbc['hit'].value_counts(),full_tsd_rfc.shape,full_tsd_rfc['hit'].value_counts())
merged_tsd_hgbc = full_tsd_hgbc.merge(top_songs_views_dataset,how="inner")
merged_tsd_rfc=full_tsd_rfc.merge(top_songs_views_dataset,how="inner")
print(merged_tsd_hgbc['hit'].value_counts(),merged_tsd_hgbc.shape,
merged_tsd_rfc['hit'].value_counts(),merged_tsd_rfc.shape)


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming merged_tsd_rfc is a DataFrame
# If it's a numpy array, you can convert it to a DataFrame using pd.DataFrame()

# Convert values to numeric
numeric_values = pd.to_numeric(merged_tsd_rfc.to_numpy().flatten(), errors='coerce')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(numeric_values)), numeric_values, alpha=0.5)
plt.title('Scatter Plot for merged_tsd_rfc')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()


# In[35]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming merged_tsd_rfc is a DataFrame
# If it's a numpy array, you can convert it to a DataFrame using pd.DataFrame()

# Convert values to numeric
numeric_values = pd.to_numeric(merged_tsd_rfc.to_numpy().flatten(), errors='coerce')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(numeric_values)), numeric_values, alpha=0.5)
plt.title('Scatter Plot for merged_msd_rfc')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()


# ## Conclusion

# In[55]:


def plot_feature_by_category(dataset, title):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=dataset.mean().reset_index(), x='index', y=0)
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)
    plt.show()

# Plot for full_hits_dataset
plot_feature_by_category(full_hits_dataset, 'Popularity of Music - Full Hits Dataset')

# Plot for top_songs_dataset
plot_feature_by_category(top_songs_dataset, 'Popularity of Music - Top Songs Dataset')

# Plot for most_songs_dataset
plot_feature_by_category(most_songs_dataset, 'Popularity of Music - Most Songs Dataset')


# In[59]:


import streamlit as st

# Assuming you have the dataframes merged_msd_rfc and merged_tsd_rfc

# Display Most Song Dataset
# st.write("Most Song Dataset")
# st.dataframe(merged_msd_rfc)

# Display Top Song Dataset
st.write("Top Song Dataset")
st.dataframe(merged_tsd_rfc)

st.title("Search in Top Song Dataset")
search_query = st.text_input("Enter a search query:")
filtered_df = merged_tsd_rfc[merged_tsd_rfc.apply(lambda row: any(search_query.lower() in str(cell).lower() for cell in row), axis=1)]
st.write("Result:")
st.dataframe(filtered_df)

# - Conclution pertanyaan 1
# - Conclution pertanyaan 2
