import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    
    ndc_dict = ndc_df.set_index('NDC_Code').to_dict()['Non-proprietary Name']
    df['generic_drug_name'] = None
    df['generic_drug_name'] = df.apply(
        lambda row: ndc_dict[row['ndc_code']] if row['ndc_code'] in ndc_dict else row['ndc_code']
        , axis=1   
    )
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.sort_values('encounter_id')
    first_encounter_values = first_encounter_df.groupby('patient_nbr')['encounter_id'].head(1).values
    first_encounter_df = first_encounter_df[first_encounter_df['encounter_id'].isin(first_encounter_values)]
    first_encounter_df.drop_duplicates(subset=['encounter_id'], inplace=True)
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    VALIDATION_PERCENT = 0.15
    TEST_PERCENT = 0.15
    
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    train_size = round(total_values * (1 - (VALIDATION_PERCENT + TEST_PERCENT) ))
    train = df[df[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    rest = df[~df[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    validation_size = round( (total_values - train_size) / 2 )
    validation = df[df[patient_key].isin(unique_values[train_size:train_size + validation_size])].reset_index(drop=True)
    test_size = total_values - (train_size + validation_size)
    test = df[df[patient_key].isin(unique_values[-test_size:])].reset_index(drop=True)
    
    print("Total number of unique patients in train = ", len(train[patient_key].unique()))
    print("Total number of unique patients in validation = ", len(validation[patient_key].unique()))
    print("Total number of unique patients in test = ", len(test[patient_key].unique()))
    print("Training partition has a shape = ", train.shape) 
    print("Validation partition has a shape = ", validation.shape) 
    print("Test partition has a shape = ", test.shape)

    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key = c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value = default_value, 
                                                          normalizer_fn=normalizer, dtype=tf.float64)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    
    df['pred_binary'] = 0
    df['pred_binary'] = df.apply(
        lambda row: 1 if row[col] >= 5 else 0
        , axis=1)
    
    student_binary_prediction = df['pred_binary'].values.tolist()
    
    
    return student_binary_prediction
