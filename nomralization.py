import pandas as pd


def normalize_dataframe(df):
    df_norm = df.copy()
    df_norm = df_norm.drop(columns='class')
    return (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())


def main():
    df = pd.read_csv (r'pima-indians-diabetes.data')
    print (df.head())
    normalized_df = normalize_dataframe(df)
    normalized_df = normalized_df.join(df["class"])
    normalized_df.to_csv('normalize_data.data', sep=',', header=False, index=False)
    
    
    
    
    
if __name__ == "__main__":
    main()