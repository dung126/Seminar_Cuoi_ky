import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Làm sạch doanh số (Scale lại về đơn vị triệu bản)
df = pd.read_csv("gamesale_clean.csv")
cols_sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
for col in cols_sales:
    # Loại bỏ dấu chấm và chia cho 10^15 để đưa về số thực nhỏ (ví dụ 36.26 triệu thay vì hàng tỷ tỷ)
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).astype(float) / 1e15

# 2. Xử lý cột Year (Tạm thời không lọc để tránh mất dữ liệu)
# Vì cột Year quá nhiễu, ta chỉ chuyển nó về dạng số, dòng nào lỗi thì để NaN
df['Year_Clean'] = pd.to_numeric(df['Year'].astype(str).str.replace('.', '', regex=False), errors='coerce')
# Thay thế năm lỗi bằng giá trị trung bình (hoặc bỏ qua cột này nếu vẫn lỗi)
df['Year_Clean'] = df['Year_Clean'].fillna(df['Year_Clean'].median())

# KIỂM TRA TRƯỚC KHI CHIA
print(f"Số lượng dòng dữ liệu hiện có: {len(df)}")

if len(df) > 0:
    # 3. Chọn đặc trưng
    X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    y = df['Global_Sales']
    # 4. Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 5. Huấn luyện Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
        # 6. Dự đoán và đánh giá
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Kết quả huấn luyện:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
else:
    print("Cảnh báo: Dữ liệu vẫn bị trống! Hãy kiểm tra lại file CSV đầu vào.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. TẢI DỮ LIỆU (Đảm bảo file 'gamesale_clean.csv' nằm cùng thư mục với file notebook)
try:
    df = pd.read_csv('gamesale_clean.csv')
    print(f"Đã tải thành công dữ liệu: {len(df)} dòng.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'gamesale_clean.csv'. Hãy kiểm tra lại tên file!")

# 2. CHỌN ĐẶC TRƯNG VÀ MỤC TIÊU
# Theo dữ liệu nguồn mới nhất: Year đã sạch, Sales đã ở dạng float thực tế
features = ['Platform', 'Genre', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
target = df.columns[-1]  # Giả định cột cuối cùng là Global_Sales

X = df[features]
y = df[target]

# 3. THIẾT LẬP TIỀN XỬ LÝ (PREPROCESSOR)
categorical_features = ['Platform', 'Genre']
numeric_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# Xử lý các giá trị thiếu cho Year và mã hóa One-Hot cho phân loại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Year']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], 
    remainder='passthrough'
)

# 4. CHIA TẬP DỮ LIỆU
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. XÂY DỰNG PIPELINE VÀ GRID SEARCH
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Thiết lập tham số tối ưu (Chỉnh sửa để không bị lỗi Got -2)
param_grid = {
    'regressor__n_estimators': [1, 2],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [3, 4]
}

print("Đang tối ưu hóa mô hình (Grid Search)... Vui lòng đợi trong giây lát.")
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 6. LẤY MÔ HÌNH TỐT NHẤT VÀ DỰ BÁO
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# 7. ĐÁNH GIÁ CHỈ SỐ (MSE, RMSE, R-squared)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rf)

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print(f"Tham số tốt nhất: {grid_search.best_params_}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# 8. VẼ BIỂU ĐỒ TRỰC QUAN HÓA
plt.figure(figsize=(12, 5))

# Biểu đồ 1: Actual vs Predicted
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title('Dự báo vs Thực tế (Global Sales)')
plt.xlabel('Thực tế (triệu bản)')
plt.ylabel('Dự báo (triệu bản)')

# Biểu đồ 2: Residuals Distribution
plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_rf, kde=True, color='green')
plt.title('Phân phối sai số (Residuals)')
plt.xlabel('Giá trị sai số')

plt.tight_layout()
plt.show()