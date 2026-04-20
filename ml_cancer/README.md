Breast Cancer Wisconsin
==============================
## Giới thiệu
Dự án này áp dụng các thuật toán Machine Learning để phân loại khối u vú thành:
- Ác tính (Malignant - M)
- Lành tính (Bengign - B)
Mục tiêu là xây dựng mô hình hỗ trợ phân tích và dự đoán trong y học.
## Mục tiêu
- Xây dựng mô hình phân loại
- Đánh giá mô hình bằng các chỉ số tiêu chuẩn.
## Dataset
- Bộ dữ liệu: Breast Cancer Dataset
- Đặc trưng: các thông số của tế bào (radius, texture,...)
- Nhãn: M(ác tính)/ B(lành tính)
## Mô hình sử dụng
Logistic Regression/ Random Forest/ SSVM
## Đánh giá mô hình
- Accuracy (độ chính xác)
- Precision
- Recall
- F1-score
- Confusion Matrix
## Công nghệ sử dụng
- Python
- Pandas
- Numpy
- Scikit-learn
## Kết quả 
Mô hình đạt độ chính xác cao trong việc phân loại

Hướng dẫn xem kết quả

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
