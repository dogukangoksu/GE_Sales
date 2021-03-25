# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:04:55 2021

@author: dogukan.goksu
"""

from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
imputer = KNNImputer(n_neighbors=2)
def impute(df, column):
    '''A function to deal with missingness based on K-Nearest Neighbors
    with number of neighbors fixed to 2. 
    Inputs: df      -> pandas dataframe object.
            column  -> Name of the column where the missingness occurs
                       should be passed as string.
    Output: df_imputed -> pandas dataframe object with imputations.
    
    Sample usage:
        df = pd.read_csv('sales_dataset.csv')
        to_delete = np.random.randint(1,105,3)
        print(to_delete)
        df['Sales'].iloc[to_delete] = np.nan
        imputed = impute(df,'Sales')
        df.isna().sum() -> 3 for Sales
        imputed.isna().sum() -> 0 for Sales
    '''
    df_imputed = df.copy()
    column_arr = np.array(df[column])
    imputed_arr = imputer.fit_transform(column_arr.reshape(-1,1))
    df_imputed[column] = imputed_arr
    return df_imputed
    
    
