Breast Cancer Wisconsin
==============================
# Phân loại ung thư vú (Breast Cancer Classification)
## Giới thiệu
Dự án này áp dụng các thuật toán Machine Learning để phân loại khối u vú thành hai loại là:
- Ác tính (Malignant - M)
- Lành tính (Bengin - B)
Mục tiêu của dự án này là xây dựng mô hình hỗ trợ phân tích và dự đoán trong y học.
# Dataset
- Bộ dữ liệu: Breast Cancer Dataset
- Đặc trưng: Các thông số của tế bào như radius, texture,...
- Nhãn : M (khối u ác tính), B (khối u lành tính)
Link dataset: https://drive.google.com/file/d/11HHk3DIuyIJqsBYknGy0_5efaJFXqjFJ/view?usp=sharing
# Mô hình sử dụng
Logistic Regression/Random Forest
# Đánh giá mô hình
Đánh giá mô hình theo:
- Accuracy
- F1-score
- Confusion matrix
# Công nghệ sử dụng
- Ngôn ngữ python
- Pandas
- Scikit-learn
# Hướng dẫn xem kết quả
- Dữ liệu đã làm sạch: `data/processed/cancer_clean.csv`
- Các biểu đồ và hình đánh giá mô hình: `reports/figures/`
- Bạn có thể xem trực tiếp các file này mà không cần chạy lại code.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
