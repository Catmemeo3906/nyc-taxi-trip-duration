# Trip Duration Prediction for New York City Taxis

Mã nguồn kèm bài báo cùng tên. Nhóm 16, môn ADY, FPT University — Ho Chi Minh City.

| Thành viên | MSSV |
|---|---|
| Tạ Dương Cát | SE203533 |
| Đinh Văn Hiếu | SE210544 |
| Quách Bảo Minh | SE210709 |

## Dữ liệu

Bộ dữ liệu là **NYC Taxi Trip Duration** của Kaggle. Kho này **không** chứa dữ liệu — tệp `train.csv` nặng khoảng 200 MB và thuộc giấy phép của cuộc thi. Tải trực tiếp từ nguồn:

<https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data>

Trên Kaggle, gắn bộ dữ liệu của cuộc thi vào notebook là xong — mã nguồn đọc
thẳng tệp `.zip`, không cần giải nén. Chạy ngoài Kaggle thì đặt `train.csv`
vào thư mục `data/`.

## Chạy lại

Hai notebook chạy **tuần tự**, notebook thứ hai đọc output của notebook thứ nhất.

| Thứ tự | Notebook | Nội dung |
|---|---|---|
| 1 | `nyc-trip-duration-cleaning-audit-and-cv.ipynb` | Làm sạch, audit chất lượng từng cột, 30 đặc trưng, so sánh 6 mô hình bằng 5-fold CV, đánh giá holdout, thí nghiệm 2×2 chia theo thời gian |
| 2 | `nyc-trip-duration-tuning-and-final-model.ipynb` | Tinh chỉnh siêu tham số, train lại trên toàn bộ development set, permutation importance, phân tích sai số, chấm mô hình trên các dòng đã bị loại, lưu mô hình |

Bản công khai kèm toàn bộ output, bấm **Copy & Edit** là chạy lại được ngay:

- <https://www.kaggle.com/code/meomeo17739/nyc-trip-duration-cleaning-audit-and-cv>
- <https://www.kaggle.com/code/meomeo17739/nyc-trip-duration-tuning-and-final-model>

Cả hai chạy được từ đầu đến cuối bằng **Restart and run all**. Biến cấu hình ở
cell đầu quyết định cỡ mẫu dùng cho so sánh và tinh chỉnh:

```python
SEED = 42
MODE = 'fast'   # 'fast' = 200.000 chuyến | 'baseline' = 300.000 | 'full' = toàn bộ
```

Kết quả trong bài báo chạy ở `MODE = 'fast'`.

## Môi trường

```
pip install -r requirements.txt
```

Chạy trên Python 3.11. Thư viện chính: pandas, numpy, scikit-learn, xgboost, lightgbm, scipy, matplotlib, duckdb.

## Kết quả chính

| | RMSLE | MAE (giây) |
|---|---|---|
| Hồi quy tuyến tính | 0,4050 | — |
| XGBoost, mẫu 200k | 0,3045 | — |
| XGBoost, toàn bộ dữ liệu, holdout | **0,2990** | **163** |
| Điểm Kaggle (nộp muộn) | 0,4199 | — |

Khoảng cách giữa holdout và điểm Kaggle là chủ đề chính của bài: các luật làm sạch loại chuyến theo chính biến mục tiêu, nên đặt một trần dự đoán vào mô hình mà tập test của cuộc thi không có.

## Tệp xuất ra

| Tệp | Nội dung |
|---|---|
| `models/best_model.joblib` | Pipeline đã huấn luyện |
| `models/model_meta.json` | Siêu dữ liệu: danh sách đặc trưng, phép biến đổi mục tiêu (`log1p` / `expm1`, đơn vị giây), chỉ số đánh giá |
| `models/baseline_comparison.csv` | Bảng so sánh 6 mô hình cơ sở |
| `models/test_results.csv` | Dự đoán trên holdout kèm phần dư |
| `figures/` | Các hình dùng trong bài |

Mô hình huấn luyện trên `log1p(trip_duration)`. Muốn ra giây thì phải `expm1` kết quả dự đoán — `model_meta.json` ghi rõ điều này.

## Giấy phép

Mã nguồn dùng cho mục đích học tập. Dữ liệu thuộc giấy phép của cuộc thi Kaggle.
