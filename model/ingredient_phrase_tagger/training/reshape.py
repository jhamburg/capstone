#!/usr/bin/env python
import pandas as pd
import os

from .cli import Cli

def reshape_data(tbl):
    """Reshape data for use in model so that each recipe is
       just one observation and the ingredients and tags are lists"""

    
    # Indexes where the sentece starts
    sentStarters = tbl.loc[tbl['index'] == 'I1']

    # Add indicator for group and fill that forward for the group
    tbl.loc[tbl['index'] == 'I1', 'sent'] = range(sentStarters.shape[0])
    tbl['sent'] = tbl['sent'].fillna(method='ffill')

    def reshape_recipe(recipe):
        tokens = [token for token in recipe['token']]
        tags = [tag for tag in recipe['tag']]
        return pd.DataFrame({'sents': [tokens], 'tags': [tags]})

    return tbl.groupby('sent').apply(reshape_recipe)
                               

def read_and_save_raw_data(dataPath, filename):
    """Read and save raw NYT data"""
    # Read in Raw data
    datafileNm = os.path.join(dataPath, 'nyt-ingredients-snapshot-2015.csv')
    nytData = pd.read_csv(datafileNm, index_col=None)
    nytData.drop(columns='index', inplace=True)

    # Generate training data from NY Times Ingredient Tagging Model
    cleaned_dat = reshape_data(Cli(nytData).df)
    cleaned_dat.to_pickle(os.path.join(dataPath, filename))
