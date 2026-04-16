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
    input_gamesale = RAW_DATA_DIR / "vgsales.csv"
    output_gamesale = PROCESSED_DATA_DIR / "gamesale_clean.csv"
    logger.info("Loading video game sales dataset.")
    df = pd.read_csv(input_gamesale)
    print(df.shape)
    print(df.head())
    print(df.info())
    #Clean
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    #Handle missing
    print(df.isnull().sum())
    #Fill numeric NaN
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    # Fill categorical NaN
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    #Normalize
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    # SAVE
    df.to_csv(output_gamesale, index=False)

    logger.success("Video game sales done!")


if __name__ == "__main__":
    main()