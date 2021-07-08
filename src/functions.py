import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder

def clean1():
    filename = "Final_MetaCyc_pathways_4347.xlsx"
    print("cleaning:", filename)
    df = pd.read_excel(config.RAW_DIR + filename, engine="openpyxl")
    indexed = df.set_index(['Study Accession', 'Sample Accession or Sample ID'])
    indexed_sorted = indexed.sort_index(level=1)
    pathways = indexed_sorted.iloc[:, 29:]
    output_dir = config.CLEAN_DIR + "pathways4347.csv"
    pathways.to_csv(output_dir)
    print("\twritten to:", output_dir) 

def clean2():
    filename = "Final_taxonomy_4347.xlsx"
    print("cleaning:", filename)
    df = pd.read_excel(config.RAW_DIR + filename, engine="openpyxl", header=None, index_col=0)
    transposed = df.T
    indexed = transposed.set_index(["Study Accession", "Sample Accession or Sample ID"])
    indexed_sorted = indexed.sort_index(level=1)
    taxonomy = indexed_sorted.iloc[:, 31:]
    taxonomy_scaled = taxonomy / 100
    output_dir = config.CLEAN_DIR + "taxonomy4347.csv"
    taxonomy_scaled.to_csv(output_dir)
    print("\twritten to:", output_dir) 
    isHealthy = indexed_sorted[['Phenotype']] == 'Healthy'
    output_dir = config.CLEAN_DIR + "isHealthy4347.csv"
    isHealthy.to_csv(output_dir)
    print("\twritten to:", output_dir)

def clean3():
    filename = "Vaidation_humann2.xlsx"
    print("cleaning:", filename)
    df = pd.read_excel(config.RAW_DIR + filename, engine="openpyxl")
    indexed = df.set_index(['Study_ID', "Sample_ID"])
    indexed_sorted = indexed.sort_index(level=1)
    pathway = indexed_sorted.iloc[:, 2:]
    output_dir = config.CLEAN_DIR + "pathways241.csv"
    pathway.to_csv(output_dir)
    print("\twritten to:", output_dir)
    isHealthy = indexed_sorted[["Phenotype"]] == 'Healthy'
    output_dir = config.CLEAN_DIR + "isHealthy241.csv"
    isHealthy.to_csv(output_dir)
    print("\twritten to:", output_dir)

def clean4():
    filename = "Validation_final_679.csv"
    print("cleaning:", filename)
    df = pd.read_csv(config.RAW_DIR + filename)
    indexed = df.set_index(['Study_ID', 'Sample_ID'])
    indexed_sorted = indexed.sort_index(level=1)
    taxonomy = indexed_sorted.iloc[:, 2:]
    taxonomy_scaled = taxonomy / 100
    output_dir = config.CLEAN_DIR + "taxonomy679.csv"
    taxonomy_scaled.to_csv(output_dir)
    print("\twrote to:", output_dir)
    isHealthy = indexed_sorted.iloc[:, [1]] == 'Healthy'
    output_dir = config.CLEAN_DIR + "isHealthy679.csv"
    isHealthy.to_csv(output_dir)
    print("\twritten to:", output_dir)

def get_groups(df):
    """
    Creates label encoded groups based on the first index of df
    """
    groups = np.asarray(list(df.index))
    first_index = groups[:, 0] # study accession
    encoder = LabelEncoder()
    encoder.fit(np.unique(first_index))
    return encoder.transform(first_index)

def load_taxonomy():
    """
    Returns X and y with just taxonomy data,
    combines both training and validation sets
    """
    taxonomy = pd.read_csv(config.CLEAN_DIR + "taxonomy4347.csv", index_col=[0,1])
    isHealthy = pd.read_csv(config.CLEAN_DIR + "isHealthy4347.csv", index_col=[0, 1])
    taxonomy_val = pd.read_csv(config.CLEAN_DIR + "taxonomy679.csv", index_col=[0, 1])
    isHealthy_val = pd.read_csv(config.CLEAN_DIR + "isHealthy679.csv", index_col=[0, 1])
    taxonomy_val_cropped = taxonomy_val[taxonomy.columns]
    X = pd.concat([taxonomy, taxonomy_val_cropped])
    y = pd.concat([isHealthy, isHealthy_val])
    return X, y

def load_both():
    """
    Returns X and y with both taxonomy and pathway data,
    combines both training and reduced validation sets
    """
    taxonomy = pd.read_csv(config.CLEAN_DIR + "taxonomy4347.csv", index_col=[0,1])
    isHealthy = pd.read_csv(config.CLEAN_DIR + "isHealthy4347.csv", index_col=[0, 1])
    taxonomy_val = pd.read_csv(config.CLEAN_DIR + "taxonomy679.csv", index_col=[0, 1])
    # elim uneccesary features
    taxonomy_val_cropped = taxonomy_val[taxonomy.columns]
    pathways = pd.read_csv(config.CLEAN_DIR + "pathways4347.csv", index_col=[0,1])
    pathways_val = pd.read_csv(config.CLEAN_DIR + "pathways241.csv", index_col=[0,1])
    taxonomy_val_cropped_reduced = taxonomy_val_cropped.loc[pathways_val.index]
    isHealthy_val_reduced = pd.read_csv(config.CLEAN_DIR + "isHealthy241.csv", index_col=[0,1])
    taxonomy_final = pd.concat([taxonomy, taxonomy_val_cropped_reduced])
    pathways_final = pd.concat([pathways, pathways_val])
    X = pd.concat([taxonomy_final, pathways_final], axis=1)
    y = pd.concat([isHealthy, isHealthy_val_reduced])
    return X, y

