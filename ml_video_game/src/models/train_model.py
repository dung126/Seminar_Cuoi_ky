import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# LÀM SẠCH DOANH SỐ (Scale lại về đơn vị triệu bản) 
try:
    df = pd.read_csv('gamesale_clean.csv')
    cols_sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    for col in cols_sales:
        # Loại bỏ dấu chấm và chia cho 10^15 để đưa về số thực nhỏ 
        df[col] = df[col].astype(str).str.replace('.', '', regex=False).astype(float) / 1e15
    print(f"Đã tải thành công dữ liệu: {len(df)} dòng.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'gamesale_clean.csv'.")

# XỬ LÝ CỘT YEAR 
# Chuyển đổi sang số, xóa dấu chấm lỗi và điền khuyết bằng trung vị
df['Year_Clean'] = pd.to_numeric(df['Year'].astype(str).str.replace('.', '', regex=False), errors='coerce')
df['Year_Clean'] = df['Year_Clean'].fillna(df['Year_Clean'].median())

# CHỌN ĐẶC TRƯNG VÀ CHIA TẬP DỮ LIỆU 
features = ['Platform', 'Genre', 'Year_Clean', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
X = df[features]
y = df['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# THIẾT LẬP TIỀN XỬ LÝ (PREPROCESSOR)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Year_Clean']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre'])
    ], remainder='passthrough'
)

# Linear Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# Random Forest Regressor 
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

# ĐÁNH GIÁ VÀ SO SÁNH KẾT QUẢ 
# Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("=== Linear Regression ===")
print("MSE:", mse_lr)
print("R2:", r2_lr)

print("\n=== Random Forest ===")
print("MSE:", mse_rf)
print("R2:", r2_rf)


plt.figure(figsize=(18, 5))

# Biểu đồ 1: Thực tế vs Dự đoán (Mô hình Linear Regression)
plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.3, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title('Dự đoán - Thực tế (Linear Regression)')
plt.xlabel('Thực tế (triệu bản)')
plt.ylabel('Dự đoán (triệu bản)')

# Biểu đồ 2: Thực tế vs Dự đoán (Mô hình Random Forest)
plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title('Dự đoán - Thực tế (Random Forest)')
plt.xlabel('Thực tế (triệu bản)')
plt.ylabel('Dự đoán (triệu bản)')

# Biểu đồ 3: Phân phối sai số (Residuals của Random Forest)
plt.subplot(1, 3, 3)
sns.histplot(y_test - y_pred_rf, kde=True, color='green')
plt.title('Phân phối sai số (Random Forest)')
plt.xlabel('Giá trị sai số')

plt.tight_layout()
plt.show()

