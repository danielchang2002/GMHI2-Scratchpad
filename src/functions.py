import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder

def load_train(boolean = False):
    """
        Returns X_train, y_train from the data directory
        If boolean, then returns X_train as a boolean matrix,
        where each nonzero value is replaced with 1
    """
    return load(config.TRAIN_DIR + "taxonomy.csv", 
            config.TRAIN_DIR + "isHealthy.csv", boolean)

def load_val(boolean = False):
    """
        Returns X_val, y_val from the data directory
        If boolean, then returns X_val as a boolean matrix,
        where each nonzero value is replaced with 1
    """
    return load(config.VAL_DIR + "taxonomy679.csv", 
            config.VAL_DIR + "isHealthy679.csv", boolean)

def load(x_dir, y_dir, boolean):
    species = pd.read_csv(x_dir, index_col=[0, 1])
    species[species < 0.00001] = 0
    X = species
    isHealthy = pd.read_csv(y_dir, index_col=[0, 1])
    y = isHealthy
    if boolean:
        X = (X > 0) * 1
    return X, y

def clean_training_taxonomy():
    """
        Messy function for cleaning excel file
    """
    filename = "Final_taxonomy_4347.xlsx"
    df = pd.read_excel(config.RAW_DIR + filename, engine="openpyxl", header=None)

    # transpose
    transposed = df.T

    # set first row as the column names
    transposed.columns = transposed.iloc[0, :]
    transposed.drop(index=0, inplace=True)

    indexed = transposed.set_index(['Study Accession', 'Sample Accession or Sample ID'])

    sorted_df = indexed.sort_index()

    # extract X and y from the indexed + sorted dataframe 

    isHealthy = sorted_df[['Phenotype']] == 'Healthy'
    isHealthy.to_csv(config.TRAIN_DIR + "isHealthy.csv")

    taxonomy = sorted_df.iloc[:, 31:]
    scaled_taxonomy = taxonomy / 100
    scaled_taxonomy.to_csv(config.TRAIN_DIR + "taxonomy.csv")


def clean_training_pathway():
    """
        Messy function for cleaning excel file
    """
    filename = "Final_MetaCyc_pathways_4347.xlsx"
    df = pd.read_excel(config.RAW_DIR + filename, engine="openpyxl")
    df2 = df.copy()
    indexed = df2.set_index(['Study Accession', 'Sample Accession or Sample ID'])
    cropped = indexed.iloc[:, 29:]
    cropped_sorted = cropped.reindex(sorted(cropped.columns), axis=1)
    cropped_sorted.to_csv(config.TRAIN_DIR + "pathways.csv")

def clean_validation_taxonomy():
    """
        Messy function for cleaning excel file
    """
    filename = "Validation_final_679.csv"
    val = pd.read_csv(
            config.RAW_DIR + filename).sort_values('Sample_ID')
    val_with_id = val.set_index(["Study_ID", "Sample_ID"])

    taxonomy_val = val_with_id.iloc[:, 2:]
    taxonomy_val_scaled = taxonomy_val / 100

    # don't use all the features, use the ones that the training set used
    X, y = load_train()
    taxonomy_val_scaled_cropped = (
            taxonomy_val_scaled[list(X.columns)])
    taxonomy_val_scaled_cropped.to_csv(config.VAL_DIR + "taxonomy679.csv")

    isHealthy_val = val_with_id.iloc[:, [1]] == 'Healthy'
    isHealthy_val.to_csv(config.VAL_DIR + "isHealthy679.csv")


def clean_validation_reduced():
    # Only 241 samples
    filename = "Vaidation_humann2.xlsx"
    val_pathways = pd.read_excel(config.RAW_DIR + filename, engine="openpyxl")
    val_pathways = val_pathways.sort_values('Sample_ID')
    val_pathways.index = val_pathways['Sample_ID']
    pathways = val_pathways.iloc[:, 4:].T

    pathways.to_csv(config.VAL_DIR + "pathways241.csv")

    isHealthy_val = pd.read_csv(config.VAL_DIR + "isHealthy679.csv", index_col=0).T

    isHealthy_val_limited = isHealthy_val[list(pathways.columns)]
    isHealthy_val_limited.to_csv(config.VAL_DIR + "isHealthy241.csv")

    val_species_scaled = pd.read_csv(
            config.VAL_DIR + "taxonomy679.csv", index_col=0) 

    species_limited = val_species_scaled.loc[list(isHealthy_val_limited.columns)]
    species_limited.T.to_csv(config.VAL_DIR + "taxonomy241.csv")

def get_groups(df):
    """
    Creates label encoded groups based on the first index of df
    """
    groups = np.asarray(list(df.index))
    first_index = groups[:, 0] # study accession
    encoder = LabelEncoder()
    encoder.fit(np.unique(first_index))
    return encoder.transform(first_index)


