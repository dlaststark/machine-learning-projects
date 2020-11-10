#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to pre-process the source data for model training

Created on Mon Nov  2 14:25:17 2020

@author: tapasdas
"""


from contextlib import redirect_stdout
from os import listdir, devnull, path
import pandas as pd
import numpy as np
import string
import pickle
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils  import to_categorical
from utils.params import lang_map, class_wght_fl, scaler_fl

with redirect_stdout(open(devnull, "w")):
    nltk.download('punkt')
    nltk.download('stopwords')


def prepare_train_data(file_path):
    """
    Function to read the raw code snippets and load into dataframe

    Parameters
    ----------
    file_path : str
        Full file path to dataset folder containing the code snippets.

    Returns
    -------
    df : dataframe
        Final dataframe containing the processed code snippets data.

    """
    
    print("[{}]    prepare_train_data module started...".format(datetime.now()))
    
    # Prepare dictionary from code snippets at input file path
    dataset = {}
    for folder in listdir(file_path):
        for idx, file in enumerate(listdir(file_path + '/' + folder)):
            f = open(file_path + '/' + folder + "/" + file, "r")
            dataset[folder + "_" + str(idx)] = f.read()
            f.close()
    
    # Convert code snippet dictionary to dataframe
    df = pd.DataFrame(list(dataset.items()), 
                      columns=['prog_lang', 'code_snippet'])
    
    # Create new ID column
    df['ID'] = df['prog_lang'].apply(lambda x: x.split('_')[1])
    df['prog_lang'] = df['prog_lang'].apply(lambda x: x.split('_')[0])
    
    # Get rid of '\n' from whitespace
    df['code_snippet'] = df['code_snippet'].apply(lambda x: x.replace('\n', ' '))
    print("[{}]    prepare_train_data module completed.".format(datetime.now()))
    return df


def prepare_predict_data(file_path):
    """
    Function to read the raw code snippets and load into dataframe

    Parameters
    ----------
    file_path : str
        Full file path to dataset folder containing the code snippets.

    Returns
    -------
    df : dataframe
        Final dataframe containing the processed code snippets data.

    """
    
    print("[{}]    prepare_predict_data module started...".format(datetime.now()))
    
    # Prepare dictionary from code snippets at input file path
    dataset = {}
    fileset = {}
    for idx, file in enumerate(listdir(file_path)):
        f = open(file_path + '/' + file, "r")
        dataset[str(idx)] = f.read()
        fileset[str(idx)] = file
        f.close()
    
    # Convert code snippets dictionary to dataframe
    df = pd.DataFrame(list(dataset.items()), 
                      columns=['ID', 'code_snippet'])
    
    # Get rid of '\n' from whitespace
    df['code_snippet'] = df['code_snippet'].apply(lambda x: x.replace('\n', ' '))
    
    print("[{}]    prepare_predict_data module completed.".format(datetime.now()))
    return df, fileset


def prepare_true_labels(df):
    """
    Function to extract and encode true labels into Numpy array

    Parameters
    ----------
    df : dataframe
        Input dataframe containing code snippets and true labels.

    Returns
    -------
    df : dataframe
        Processed dataframe after removing true labels.
    train_y : Numpy array
        Array containing encoded true labels.

    """
    
    print("[{}]    prepare_true_labels module started...".format(datetime.now()))
    
    df['prog_lang'] = df['prog_lang'].map(lang_map)
    train_y = np.array([df['prog_lang'].values]).T
    df.drop(['prog_lang'], inplace=True, axis=1)
    print("[{}]    prepare_true_labels module completed.".format(datetime.now()))
    return df, train_y


def prepare_class_weights(train_y):
    """
    Function to calculate the weights for individual classes in source dataset

    Parameters
    ----------
    train_y : Numpy array
        Array containing encoded true labels.

    Returns
    -------
    class_weight : dict
        Dictionary containing weights for individual classes.

    """
    
    print("[{}]    prepare_class_weights module started...".format(datetime.now()))
    
    temp_df = pd.DataFrame(train_y, columns=['target'])
    temp_df = temp_df.groupby(['target']).size().reset_index().rename(columns={0:'count'})
    total_count = np.sum(temp_df['count'].values)
    temp_df['class%'] = (temp_df['count'] / total_count) * 100
    lowest_pct = min(temp_df['class%'])
    temp_df['class_weight'] = lowest_pct / temp_df['class%']
    class_weight = temp_df[['target', 'class_weight']].to_dict()['class_weight']
    
    file = open(class_wght_fl, 'wb')
    pickle.dump(class_weight, file)
    file.close()
    
    print("[{}]    prepare_class_weights module completed.".format(datetime.now()))
    return class_weight


def feature_engg(df, embed_file):
    """
    Function to perform feature engineering on source dataset

    Parameters
    ----------
    df : dataframe
        Input dataframe containing code snippets.
    embed_file : str
        Full file path to folder containing saved models for sentence embeddings.

    Returns
    -------
    df : dataframe
        Processed dataframe after feature engineering.

    """
    
    print("[{}]    feature_engg module started...".format(datetime.now()))
    
    #--------------------------------------------
    #         Basic feature engineering
    #--------------------------------------------
    
    stop = stopwords.words('english')
    df["code_snippet_num_words"] = df["code_snippet"].apply(lambda x: len(str(x).split()))
    df["code_snippet_num_unique_words"] = df["code_snippet"].apply(lambda x: len(set(str(x).split())))
    df["code_snippet_num_chars"] = df["code_snippet"].apply(lambda x: len(str(x)))
    df["code_snippet_num_stopwords"] = df["code_snippet"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop]))
    df["code_snippet_num_punctuations"] = df['code_snippet'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['code_snippet_num_hastags'] = df['code_snippet'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))
    df['code_snippet_num_@'] = df['code_snippet'].apply(lambda x: len([x for x in str(x).split() if x.startswith('@')]))
    df['code_snippet_num_$'] = df['code_snippet'].apply(lambda x: len([x for x in str(x).split() if x.startswith('$')]))
    df['code_snippet_num_numerics'] = df['code_snippet'].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
    df["code_snippet_num_words_upper"] = df["code_snippet"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df["code_snippet_num_words_title"] = df["code_snippet"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    
    #--------------------------------------------
    #            TF-IDF Vectorization
    #--------------------------------------------
    print("[{}]      TF-IDF Vectorization started...".format(datetime.now()))
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 10), max_features=500)
    features = tfidf.fit_transform(df.code_snippet).toarray()
    features_df = pd.DataFrame(features, columns=tfidf.get_feature_names())
    df = pd.merge(df, features_df, left_index=True, right_index=True)
    print("[{}]      TF-IDF Vectorization completed.".format(datetime.now()))
    
    #--------------------------------------------
    #        Get sentence embedding models
    #--------------------------------------------
    if path.isfile(embed_file):
        with open(embed_file, 'rb') as handle: 
            data = handle.read()
        
        embed_models = pickle.loads(data)
        roberta_embedder = embed_models['roberta_embedder']
        bert_embedder = embed_models['bert_embedder']
    
    else:
        print("[{}]      Loading Roberta Base model...".format(datetime.now()))
        roberta_embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        
        print("[{}]      Loading BERT Base model...".format(datetime.now()))
        bert_embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        
        # Save the models for reusability
        embed_models = {}
        embed_models['roberta_embedder'] = roberta_embedder
        embed_models['bert_embedder'] = bert_embedder
        
        file = open(embed_file, 'wb')
        pickle.dump(embed_models, file)
        file.close()
    
    #--------------------------------------------
    #           Roberta Base Embedding
    #--------------------------------------------
    print("[{}]      Roberta Base Embedding started...".format(datetime.now()))
    roberta_sentence_embeddings = roberta_embedder.encode(df.code_snippet.values.tolist(), batch_size=64, show_progress_bar=True)
    features_df = pd.DataFrame(roberta_sentence_embeddings)
    df = pd.merge(df, features_df, left_index=True, right_index=True)
    print("[{}]      Roberta Base Embedding completed.".format(datetime.now()))
    
    #--------------------------------------------
    #            BERT Base Embedding
    #--------------------------------------------
    print("[{}]      BERT Base Embedding started...".format(datetime.now()))
    bert_sentence_embeddings = bert_embedder.encode(df.code_snippet.values.tolist(), batch_size=64, show_progress_bar=True)
    features_df = pd.DataFrame(bert_sentence_embeddings)
    df = pd.merge(df, features_df, left_index=True, right_index=True)
    print("[{}]      BERT Base Embedding completed.".format(datetime.now()))
    
    # Drop redundant columns after feature engineering
    df.drop(['ID','code_snippet'], inplace=True, axis=1)
    print("[{}]    feature_engg module completed.".format(datetime.now()))
    return df


def train_test_split(train_x, train_y):
    """
    Function to split source data into train/validation/test datasets

    Parameters
    ----------
    train_x : Numpy array
        Array containing features for code snippets.
    train_y : TYPE
        Array containing encoded true labels.

    Returns
    -------
    data_dict : dict
        Dictionary containing train/validation/test datasets.

    """
    
    print("[{}]    train_test_split module started...".format(datetime.now()))
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
    for train_index, test_index in sss.split(train_x, train_y):
        Xtrain, Xtest = train_x[train_index], train_x[test_index]
        Ytrain, Ytest = train_y[train_index], train_y[test_index]
    
    # Convert feature matrices to 3-D tensor for CNN model training
    Xtrain = np.expand_dims(Xtrain, axis=2)
    Xtest = np.expand_dims(Xtest, axis=2)
    
    # Convert true label matrices to one-hot encoding
    num_classes = len(np.unique(train_y))
    Ytrain_oh = to_categorical(Ytrain, num_classes)
    
    data_dict = {}
    data_dict['Xtrain'] = Xtrain
    data_dict['Ytrain'] = Ytrain
    data_dict['Ytrain_oh'] = Ytrain_oh
    data_dict['Xtest'] = Xtest
    data_dict['Ytest'] = Ytest
    
    print("[{}]    train_test_split module completed.".format(datetime.now()))
    return data_dict


def store_train_data(file_name, data_dict):
    """
    Function save the training data for reusability

    Parameters
    ----------
    file_name : str
        Full path and file name for saving training data.
    data_dict : dict
        Dictionary containing train/validation/test datasets.

    Returns
    -------
    None.

    """
    
    print("[{}]    store_train_data module started...".format(datetime.now()))
    
    Xtrain = data_dict['Xtrain']
    Ytrain = data_dict['Ytrain']
    Ytrain_oh = data_dict['Ytrain_oh']
    Xtest = data_dict['Xtest']
    Ytest = data_dict['Ytest']
    
    np.savez_compressed(file_name,
                        Xtrain=Xtrain, Ytrain=Ytrain, 
                        Ytrain_oh=Ytrain_oh, 
                        Xtest=Xtest, Ytest=Ytest)
    
    print("[{}]    store_train_data module completed.".format(datetime.now()))


def load_train_data(file_name):
    """
    Function to load training data from NPZ file

    Parameters
    ----------
    file_name : str
        Full path and file name for NPZ file.

    Returns
    -------
    data_dict : dict
        Dictionary containing train/validation/test datasets.

    """
    
    print("[{}]    load_train_data module started...".format(datetime.now()))
    
    processed_dataset = np.load(file_name, allow_pickle=True)
    Xtrain, Ytrain = processed_dataset['Xtrain'], processed_dataset['Ytrain']
    Ytrain_oh = processed_dataset['Ytrain_oh']
    Xtest, Ytest = processed_dataset['Xtest'], processed_dataset['Ytest']
    
    data_dict = {}
    data_dict['Xtrain'] = Xtrain
    data_dict['Ytrain'] = Ytrain
    data_dict['Ytrain_oh'] = Ytrain_oh
    data_dict['Xtest'] = Xtest
    data_dict['Ytest'] = Ytest
    
    print("[{}]    load_train_data module completed.".format(datetime.now()))
    return data_dict


def preprocess_data(dataset_path, train_pred_flag, npz_file, embed_file):
    """
    Function to pre-process the source data for model training

    Parameters
    ----------
    dataset_path : str
        Full file path to dataset folder containing the code snippets.
    train_pred_flag : str
        Flag to switch preprocessing between train/test datasets.
    npz_file : str
        Full file path to folder containing saved dataset file.
    embed_file : str
        Full file path to folder containing saved models for sentence embeddings.

    Returns
    -------
    data_dict : TYPE
        DESCRIPTION.
    class_weight : TYPE
        DESCRIPTION.

    """
    
    if train_pred_flag.upper() == 'TRAIN':
        
        print("[{}]  Starting data preprocessing for training data...".format(datetime.now()))
        
        if not(path.isfile(npz_file)):
            
            # Pre-process data
            df = prepare_train_data(dataset_path)
            df, train_y = prepare_true_labels(df)
            class_weight = prepare_class_weights(train_y)
            df = feature_engg(df, embed_file)
            
            # Normalize data
            train_x = df.values
            scaler = MinMaxScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            
            # Save the scaler object for reusability
            scaler_obj = {}
            scaler_obj['scaler'] = scaler
            
            file = open(scaler_fl, 'wb')
            pickle.dump(scaler_obj, file)
            file.close()
            
            # Split data into train/validation/test datasets
            data_dict = train_test_split(train_x, train_y)
            
            # Save training data for reusability
            store_train_data(npz_file, data_dict)
        
        else:
            
            # Load training data from NPZ file
            data_dict = load_train_data(npz_file)
            
            # Read class weights
            with open(class_wght_fl, 'rb') as handle: 
                data = handle.read()
            
            class_weight = pickle.loads(data)
        
        print("[{}]  Data preprocessing for training data completed.".format(datetime.now()))
        return data_dict, class_weight
    
    elif train_pred_flag.upper() == 'PREDICT':
        
        print("[{}]  Starting data preprocessing for test data...".format(datetime.now()))
        
        # Pre-process data
        df, fileset = prepare_predict_data(dataset_path)
        df = feature_engg(df, embed_file)
        
        # Normalize data
        with open(scaler_fl, 'rb') as handle: 
            data = handle.read()
        
        scaler_obj = pickle.loads(data)
        scaler = scaler_obj['scaler']
        
        predict_x = df.values
        predict_x = scaler.transform(predict_x)
        data_dict = {}
        data_dict['Xpredict'] = predict_x
        
        # Read class weights
        with open(class_wght_fl, 'rb') as handle: 
            data = handle.read()
        
        class_weight = pickle.loads(data)
        
        print("[{}]  Data preprocessing for test data completed.".format(datetime.now()))
        return data_dict, class_weight, fileset
