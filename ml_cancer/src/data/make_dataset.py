from pathlib import Path
import pandas as pd
from loguru import logger 
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_cancer = RAW_DATA_DIR / "data.csv"
    output_cancer = PROCESSED_DATA_DIR / "cancer_clean.csv"
    logger.info("Loading breast cancer dataset.")
    df = pd.read_csv(input_cancer)
    print(df.shape)
    print(df.head())
    print(df.info())
    #Clean
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    print(df.isnull().sum())
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    # NORMALIZE
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    # SAVE
    df.to_csv(output_cancer, index=False)

    logger.success("Breast cancer done!")


if __name__ == "__main__":
    main()