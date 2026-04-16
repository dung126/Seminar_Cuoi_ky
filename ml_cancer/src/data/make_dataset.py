from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main():
    input_cancer = RAW_DATA_DIR / "data.csv"
    output_cancer = PROCESSED_DATA_DIR / "cancer_clean.csv"
    logger.info("Loading breast cancer dataset.")
    df = pd.read_csv(input_cancer)
    print("Các cột:", df.columns)
    print(df.head())

    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # Tiền xử lý số
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # MÃ HÓA NHÃN diagnosis: M->1, B->0
    print('Giá trị diagnosis trước mã hóa:', df['diagnosis'].unique())
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    print('Giá trị diagnosis sau mã hóa:', df['diagnosis'].unique())

    # Scale feature số, bỏ cột id & diagnosis
    feature_cols = [col for col in df.select_dtypes(include=["int64", "float64"]).columns if col not in ['id', 'diagnosis']]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Ghi đè file cũ
    df.to_csv(output_cancer, index=False)
    logger.success(f"Clean data saved to {output_cancer}")

if __name__ == "__main__":
    main()