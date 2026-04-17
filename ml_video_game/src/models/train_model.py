import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  # THÊM MỚI: Thư viện Linear Regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# [1] 1. Làm sạch doanh số (Scale lại về đơn vị triệu bản)
try:
    df = pd.read_csv('gamesale_clean.csv') # [2]
    cols_sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    for col in cols_sales:
        # Loại bỏ dấu chấm và chia cho 10^15 để đưa về số thực nhỏ [1]
        df[col] = df[col].astype(str).str.replace('.', '', regex=False).astype(float) / 1e15
    print(f"Đã tải thành công dữ liệu: {len(df)} dòng.") # [2]
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'gamesale_clean.csv'. Hãy kiểm tra lại tên file!") # [2]

# [3] 2. Xử lý cột Year
df['Year_Clean'] = pd.to_numeric(df['Year'].astype(str).str.replace('.', '', regex=False), errors='coerce')
df['Year_Clean'] = df['Year_Clean'].fillna(df['Year_Clean'].median()) # [3]

# [2], [4] 3. Chọn đặc trưng và Mục tiêu
features = ['Platform', 'Genre', 'Year_Clean', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
target = df.columns[-1] 
X = df[features]
y = df[target]

# [4] 4. Thiết lập tiền xử lý (Preprocessor)
categorical_features = ['Platform', 'Genre']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Year_Clean']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough'
)

# [4] 5. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- THÊM MỚI: HUẤN LUYỆN LINEAR REGRESSION ---
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
# ----------------------------------------------

# [5] 6. Xây dựng Pipeline Random Forest và Grid Search
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Thiết lập tham số tối ưu từ nguồn [5]
param_grid = {
    'regressor__n_estimators': [100, 200, 300],  # Tăng số lượng cây
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [5, 6]
}

print("Đang tối ưu hóa mô hình Random Forest (Grid Search)...")
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# [6] 7. Lấy mô hình tốt nhất và dự báo
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# 8. Đánh giá và So sánh [6]
def print_evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- KẾT QUẢ {model_name} ---")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

print_evaluate(y_test, y_pred_lr, "LINEAR REGRESSION")
print_evaluate(y_test, y_pred_rf, "RANDOM FOREST (OPTIMIZED)")
print(f"Tham số tốt nhất RF: {grid_search.best_params_}") # [6]

# [7] 9. Vẽ biểu đồ trực quan hóa
plt.figure(figsize=(15, 5))

# Biểu đồ 1: Thực tế vs Dự báo (Của mô hình RF tốt nhất) [7]
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, color='blue', label='Random Forest')
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.3, color='orange', label='Linear Regression') # Thêm LR vào biểu đồ
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title('Dự báo vs Thực tế (Global Sales)')
plt.xlabel('Thực tế (triệu bản)')
plt.ylabel('Dự báo (triệu bản)')
plt.legend()

# Biểu đồ 2: Phân phối sai số [7]
plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_rf, kde=True, color='green')
plt.title('Phân phối sai số (Residuals - Random Forest)')
plt.xlabel('Giá trị sai số')

plt.tight_layout()
plt.show()