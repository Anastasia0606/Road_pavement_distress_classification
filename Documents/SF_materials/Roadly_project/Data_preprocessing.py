import zipfile
from glob import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from pathlib import Path
import seaborn as sns



#creating the list of paths to json files in directory
path_to_json_files = 'meta'
filepaths = glob(path_to_json_files + '/*.json') # get list of json files in folder
len(filepaths) # get the number of json files

# Creating an annotation CSV file
#Get dataframe from multiple json files, create 'file_name' column from json file name.
import json
combined_results = []
for file in filepaths:
    f = open(file, 'r')
    combined_results.append(json.load(f))

pd.DataFrame(combined_results)

meta_df = pd.DataFrame(data = pd.read_json(filepaths[0], typ='series')).transpose()
meta_df['file_name'] = os.path.basename(filepaths[0]).rsplit('.', 1)[0]

for f in filepaths[1:]:
    df = pd.DataFrame(data = pd.read_json(f, typ='series')).transpose()
    df['file_name'] = os.path.basename(f).rsplit('.', 1)[0]
    meta_df = pd.concat([meta_df, df], ignore_index=True)

meta_df.info()

#Save meta_df in csv format.
meta_df.to_csv('meta.csv')

# Compare image file names and metadata files
#loading meta data from csv, check for duplicates

meta_df = pd.read_csv('meta.csv')
data_df = meta_df.copy()
meta_df.head(3)
print(len(data_df['file_name']))
d = data_df.drop_duplicates(subset = 'file_name', ignore_index=True) # check for duplicates
len(d['file_name'])

#getting the list of image names, checking for duplicates, for example we used medium padding files 
path_to_images = 'crops/medium'
filepaths_img = glob(path_to_images + '/*.png') # get list of png files in folder
print(len(filepaths_img)) # get the number of png files
file_names_img = [os.path.basename(img_path) for img_path in filepaths_img]
file_names_img_dupldrop = list(set(file_names_img)) # check for duplicates
print(len(file_names_img_dupldrop))

#checking for meta data files that don't correspond to the image files and 
#remove them from the dataframe
data_df['file_name'] = data_df['file_name'].apply(lambda x: x + '.png')
data_df['file_name'] = data_df['file_name'].apply(lambda x: x if x in file_names_img else 'absent_img')
data_df = data_df[data_df.file_name != 'absent_img']
data_df.index = range(len(data_df))
data_df.info()

#saving new dataframe to csv for using it as annotation file
data_df.to_csv('meta_medium.csv')

#Data analysis
#checking for class balance

meta_medium_df = pd.read_csv('meta_medium.csv')
data_df = meta_medium_df.copy()
data_df.columns.values
data_df.drop(columns = ['Unnamed: 0'], inplace = True)
data_df.info()
class_df = data_df.groupby(['class_label']).agg({
    'job':'nunique',
    'validated':'sum',
    'file_name': 'count'
}).reset_index().rename(columns={'job': 'job_count', 'validated': 'validated_count', 'file_name': 'file_count'})
class_df['validation_percent'] = class_df['validated_count']/class_df['file_count']*100

# create a barplot with data distribution through the classes
fig, axs = plt.subplots(3, sharex=True)
sns.barplot(data=class_df, x='class_label', y='job_count', ax= axs[0])
sns.barplot(data=class_df, x='class_label', y='file_count', ax= axs[1])
sns.barplot(data=class_df, x='class_label', y='validation_percent', ax= axs[2])

fig.suptitle('Data distribution through the classes')
axs[2].xaxis.set_tick_params(rotation=45)
fig.tight_layout()

#creating validation dataset on images that were validated
validated_df = data_df[data_df.validated == 1]

#some classes are very scarce and 'crack_reflection' class is absent in validated_df, 
#so we added more data on this classes from unvalid images

cr_refl_df=data_df[data_df.class_label == 'crack_reflection'].groupby('job').agg({'validated':'sum',
    'file_name': 'count'}
).reset_index().rename(columns={'validated': 'validated_count', 'file_name': 'file_count'})
cr_refl_df.sort_values(by = 'file_count', ascending=False).head()

# we choose job 3384 and 3352 to enlarge validation data
val_df = pd.concat([validated_df, data_df[data_df.job == 3384], data_df[data_df.job == 3352]])

class_df1 = val_df.groupby(['class_label']).agg({
    'job':'nunique',
    'validated':'sum',
    'file_name': 'count'
}).reset_index().rename(columns={'job': 'job_count', 'validated': 'validated_count', 'file_name': 'file_count'})
class_df1

# create the barplot for distribution through the classes in validation data
fig, axs = plt.subplots(2, sharex=True)
sns.barplot(data=class_df1, x='class_label', y='job_count', ax= axs[0])
sns.barplot(data=class_df1, x='class_label', y='file_count', ax= axs[1])
fig.suptitle('Distribution through the classes in validation data')
axs[1].xaxis.set_tick_params(rotation=45)
fig.tight_layout()


# create train data from the rest of images, so we droped the validation data using list of indexes
val_index = list(val_df.index)
test_df = data_df.drop(axis = 0, index = val_index)

class_df2 = test_df.groupby(['class_label']).agg({
    'job':'nunique',
    'validated':'sum',
    'file_name': 'count'
}).reset_index().rename(columns={'job': 'job_count', 'validated': 'validated_count', 'file_name': 'file_count'})
class_df1

# create the barplot for distribution through the classes in train data
fig, axs = plt.subplots(2, sharex=True)
sns.barplot(data=class_df2, x='class_label', y='job_count', ax= axs[0])
sns.barplot(data=class_df2, x='class_label', y='file_count', ax= axs[1])
fig.suptitle('Distribution through the classes in train data')
axs[1].xaxis.set_tick_params(rotation=45)
fig.tight_layout()


# testing for data leakage, check if test and validation dataset have common jobs
val_job_list = list(set(val_df['job']))
test_job_list = list(set(test_df['job']))

def test_data_leakage(test_list, val_list):
    mask_lst = [x in test_list for x in val_list]
    return sum(mask_lst)

test_data_leakage(test_job_list, val_job_list)


# create test and val csv files
test_df.to_csv("train_data_csv")
val_df.to_csv("val_data_csv")


# since the train df was too big and unbalanced, 
# we shrank it by removing images from the most common classes

train_df = pd.read_csv('train_data_csv')

def get_index_to_remove(df):
    
    label_list = ['crack_longitudinal', 'crack_transversal', ]
    index_to_remove = []
    for label in label_list:
        for job in df[df.class_label == label].job:
            mask = (df.class_label == label) & (df.job == job)
            if len(df[mask])>30:
                index_list = df[mask].index
                index_to_remove.extend(index_list[30:])
            else:
                continue
    return index_to_remove  

df = train_df.copy()
remove_lst = get_index_to_remove(df)
df.drop(index = remove_lst, inplace = True)
len(df) 
# we reduced number of images from 39882 to 21064
df.to_csv('small_train_csv')


class_df3 = df.groupby(['class_label']).agg({
    'job':'nunique',
    'validated':'sum',
    'file_name': 'count'
}).reset_index().rename(columns={'job': 'job_count', 'validated': 'validated_count', 'file_name': 'file_count'})

# create the barplot for distribution through the classes in reduced train data
fig, axs = plt.subplots(2, sharex=True)
sns.barplot(data=class_df3, x='class_label', y='job_count', ax= axs[0])
sns.barplot(data=class_df3, x='class_label', y='file_count', ax= axs[1])
fig.suptitle('Distribution through the classes in reduced train data')
axs[1].xaxis.set_tick_params(rotation=45)
fig.tight_layout()
