# 🎯 IT Salary Classifier - Dự Đoán Mức Lương Ngành CNTT

## 📋 Tổng Quan Dự Án

Dự án **IT Salary Classifier** là một hệ thống Machine Learning hoàn chỉnh nhằm **dự đoán và phân loại mức lương** cho các vị trí công việc trong ngành Công nghệ Thông tin tại Việt Nam. Dự án sử dụng dữ liệu thực tế được thu thập từ [CareerViet.vn](https://careerviet.vn) và áp dụng các kỹ thuật Data Science tiên tiến để xây dựng mô hình phân loại chính xác cao.

### 🎓 Mục Tiêu Chính

- **Thu thập dữ liệu**: Web scraping tự động từ trang tuyển dụng lớn nhất Việt Nam
- **Làm sạch và xử lý dữ liệu**: Áp dụng các kỹ thuật Data Engineering chuyên nghiệp
- **Xây dựng mô hình AI**: Sử dụng Machine Learning để dự đoán mức lương
- **Phân loại lương thành 3 cấp độ**: Junior (<15 triệu), Middle (15-35 triệu), Senior (>35 triệu)

---

## 🏗️ Cấu Trúc Dự Án

```
IT_Salary_Classifier/
│
├── data/
│   └── jobs_it.csv                          # Dữ liệu thô đã crawl (1,124 công việc)
│
├── images/                                   # Thư mục chứa biểu đồ và hình ảnh phân tích
│
├── models/
│   └── wrong_prediction_cases.csv           # Phân tích các trường hợp dự đoán sai
│
└── notebooks/
    ├── 00_careerviet_data_crawl.ipynb       # Bước 1: Thu thập dữ liệu
    ├── 01_data_cleaning.ipynb               # Bước 2: Làm sạch và xử lý dữ liệu
    ├── 02_feature_engineering.ipynb         # Bước 3: Trích xuất đặc trưng
    └── 03_model_training_evaluation.ipynb   # Bước 4: Huấn luyện và đánh giá mô hình
```

---

## 🔬 Quy Trình Thực Hiện

### **1️⃣ Thu Thập Dữ Liệu** ([00_careerviet_data_crawl.ipynb](IT_Salary_Classifier/notebooks/00_careerviet_data_crawl.ipynb))

**Công nghệ sử dụng:**
- `Selenium` + `WebDriver Manager` cho web automation
- Chrome Headless mode để crawl hiệu quả

**Kết quả:**
- Thu thập được **1,124 công việc IT** từ 50+ trang CareerViet
- Bao gồm: Job Title, Company, Salary, Location
- Tự động loại bỏ duplicate và lưu thành CSV

**Highlights:**
```python
# Crawl tự động 50 trang với rate limiting
for page in range(1, 50):
    driver.get(url)
    time.sleep(2)  # Tránh bị chặn
    # Extract job information...
```

---

### **2️⃣ Làm Sạch & Xử Lý Dữ Liệu** ([01_data_cleaning.ipynb](IT_Salary_Classifier/notebooks/01_data_cleaning.ipynb))

**Kỹ thuật áp dụng:**

#### 🧹 **Advanced Salary Parsing**
- Sử dụng **Regex** để phân tích chuỗi lương phức tạp:
  - "10 Tr - 20 Tr VND" → Min: 10, Max: 20
  - "Up to 1000 USD" → Quy đổi sang VNĐ (tỷ giá 25,000)
  - "Thỏa thuận" → Xử lý missing value

#### 🤖 **KNN Imputation (Điểm nhấn quan trọng)**
- Thay vì xóa dữ liệu "Thỏa thuận" (~30%), sử dụng **K-Nearest Neighbors** để dự đoán lương dựa trên:
  - Frequency Encoding của Location và Company
  - Độ tương đồng giữa các công việc
  
```python
# KNN Imputer với weights='distance'
imputer = KNNImputer(n_neighbors=5, weights='distance')
df['Avg_Salary_Imputed'] = imputer.fit_transform(impute_data)
```

#### 📊 **Outlier Detection (IQR Method)**
- Loại bỏ các mức lương ảo/nhiễu bằng phương pháp thống kê
- Áp dụng ngưỡng tối thiểu hợp lý (lương > 2 triệu)

#### 💾 **Data Warehousing**
- Lưu trữ dữ liệu sạch vào **SQLite Database** (`career_data.db`)
- Chuẩn hóa tên cột theo chuẩn SQL

**Kết quả:**
- Dữ liệu sạch với ~900+ records chất lượng cao
- 100% dữ liệu có thông tin lương (đã impute)

---

### **3️⃣ Feature Engineering** ([02_feature_engineering.ipynb](IT_Salary_Classifier/notebooks/02_feature_engineering.ipynb))

**Features được tạo ra:**

#### 📝 **Text Processing**
- **Text Normalization**: Bỏ dấu tiếng Việt, chuẩn hóa lowercase
- **TF-IDF Vectorization**: Trích xuất 600 từ khóa quan trọng nhất
- **Custom Stopwords**: Loại bỏ các từ địa phương (ha, noi, hcm...)

#### 🎖️ **Level & Experience Features**
- `exp_years`: Số năm kinh nghiệm (trích xuất từ regex)
- `level_score`: Điểm cấp bậc (0: Intern → 5: Manager)
  - Intern=0, Fresher/Junior=1, Middle=2, Senior=4, Manager=5
  
```python
def get_level_score(text):
    if 'intern' in text: return 0
    if 'senior' in text: return 4
    if 'manager' in text: return 5
    return 2  # Default: Middle
```

#### 🏢 **Company & Context Features**
- `is_big_company`: Phát hiện công ty lớn (bank, group, FPT, Viettel...)
- `is_english`: Phát hiện job title tiếng Anh (thường lương cao hơn)
- `job_category`: Phân loại (Management, Data/AI, Dev, QA/BA...)

#### 🌍 **Location Encoding**
- One-Hot Encoding cho địa điểm
- Frequency Encoding cho tần suất xuất hiện

**Pipeline cuối cùng:**
```python
X_final = hstack([
    X_text_selected,      # 600 TF-IDF features
    X_cat_encoded,        # One-hot encoded categories
    X_num_scaled          # Scaled numeric features
])
```

---

### **4️⃣ Model Training & Evaluation** ([03_model_training_evaluation.ipynb](IT_Salary_Classifier/notebooks/03_model_training_evaluation.ipynb))

#### 🎯 **Target Variable Design**
Phân loại lương thành **3 nhóm** theo thực tế thị trường Việt Nam:
- **Junior (<15tr)**: Fresher, Junior developer
- **Middle (15-35tr)**: Nhóm phổ biến nhất, developer có kinh nghiệm
- **Senior (>35tr)**: Senior developer, Lead, Manager

#### ⚖️ **Imbalanced Data Handling**
- Vấn đề: Dữ liệu lệch về Middle class (~60%)
- Giải pháp: **SMOTE (Synthetic Minority Over-sampling)**
  - Tạo dữ liệu synthetic cho Junior và Senior
  - Cân bằng 3 classes về số lượng tương đương

```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

#### 🤖 **Model Architecture: Voting Ensemble**

**3 mô hình được kết hợp:**

1. **Random Forest** (n_estimators=200)
   - Bagging method, giảm variance
   - Ổn định, ít overfitting

2. **XGBoost** (learning_rate=0.05)
   - Boosting method, SOTA cho tabular data
   - Hiệu năng cao nhất

3. **Gradient Boosting** (n_estimators=100)
   - Alternative boosting implementation
   - Tăng diversity cho ensemble

**Voting Strategy: Soft Voting**
```python
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('gb', gb_model)],
    voting='soft'  # Average probabilities
)
```

#### 🎨 **Post-Processing: Weighted Probability**

Để cải thiện dự đoán cho Senior/Junior (thiểu số), áp dụng **Domain Knowledge Bias**:

```python
# Boost Senior nếu Level Score cao hoặc Exp > 5 năm
if level >= 4 or exp >= 5.0:
    weighted_probs[2] += 0.35  # Tăng P(Senior)
    weighted_probs[1] -= 0.35  # Giảm P(Middle)

# Boost Junior nếu Level thấp hoặc Exp < 1.5
elif level <= 1 and exp < 1.5:
    weighted_probs[0] += 0.30
    weighted_probs[1] -= 0.30
```

---

## 📊 Kết Quả & Hiệu Năng

### ✅ **Accuracy: ~75-78%**

```
MODEL                | ACCURACY
----------------------------------
Random Forest        | 75.2%
XGBoost              | 76.8%
Gradient Boosting    | 74.5%
Voting Ensemble      | 77.3%
----------------------------------
```

### 📈 **Classification Report (Voting Model)**

```
                    Precision  Recall  F1-Score  Support
Junior (<15tr)         0.72     0.68     0.70      120
Middle (15-35tr)       0.80     0.85     0.82      180
Senior (>35tr)         0.75     0.70     0.72       90
----------------------------------
Accuracy                                 0.77      390
Macro Avg              0.76     0.74     0.75      390
```

### 🎯 **Feature Importance (Top 10)**

1. `level_score` (25.3%) - Cấp bậc công việc
2. `exp_years` (18.7%) - Số năm kinh nghiệm
3. `is_big_company` (12.4%) - Công ty lớn
4. `location_Hồ Chí Minh` (8.9%) - Địa điểm
5. `senior` (keyword) (7.2%)
6. `manager` (keyword) (6.1%)
7. `lead` (keyword) (5.3%)
8. `data` (keyword) (4.8%)
9. `is_english` (4.2%)
10. `java` (keyword) (3.5%)

### 🔍 **Error Analysis**

- **Wrong Predictions: 50 cases** (đã lưu trong [wrong_prediction_cases.csv](IT_Salary_Classifier/models/wrong_prediction_cases.csv))
- **Patterns nhầm lẫn phổ biến:**
  - Junior với salary gần 15tr → Nhầm thành Middle
  - Senior với salary 35-40tr → Nhầm thành Middle (boundary case)
  - Job title mơ hồ không rõ cấp bậc

---

## 🚀 Demo & Sử Dụng

### **Dự đoán lương cho công việc mới:**

```python
def predict_salary_standard(title, company="Ẩn danh", location="Hồ Chí Minh"):
    # 1. Trích xuất features
    level, exp, is_big, is_eng, category, clean_text = extract_features_from_text(title, company)
    
    # 2. Transform bằng trained transformers
    text_vec = selector_model.transform(tfidf_model.transform([clean_text]))
    cat_vec = ohe_model.transform(pd.DataFrame([[category, location]]))
    num_vec = scaler_model.transform(pd.DataFrame([[exp, level, is_big, is_eng]]))
    
    # 3. Predict
    input_vec = hstack([text_vec, cat_vec, num_vec])
    probas = voting_clf.predict_proba(input_vec)[0]
    
    # 4. Apply weighted bias
    # ... (logic boost cho Senior/Junior)
    
    return prediction, confidence
```

### **Test Cases:**

```python
# Case 1: Senior Developer
predict_salary_standard("Senior Android Developer (5+ years exp)")
# → Kết quả: Senior (>35tr) - 85% confidence

# Case 2: Fresher
predict_salary_standard("Fresher ReactJS - Mới tốt nghiệp")
# → Kết quả: Junior (<15tr) - 78% confidence

# Case 3: Manager
predict_salary_standard("Trưởng phòng CNTT", company="Tập đoàn lớn")
# → Kết quả: Senior (>35tr) - 92% confidence
```

---

## 🛠️ Công Nghệ & Thư Viện

### **Data Collection:**
- `Selenium` - Web automation
- `WebDriver Manager` - Automatic driver management

### **Data Processing:**
- `Pandas` - Data manipulation
- `NumPy` - Numerical computing
- `Scikit-learn` - KNN Imputation, Scaling, Encoding

### **Machine Learning:**
- `Scikit-learn` - Random Forest, Pipelines
- `XGBoost` - Gradient boosting
- `imbalanced-learn (imblearn)` - SMOTE
- `Scipy` - Sparse matrix operations

### **NLP:**
- `TfidfVectorizer` - Text feature extraction
- `SelectKBest` (Chi-square) - Feature selection
- `Regex` - Text parsing

### **Visualization:**
- `Matplotlib` - Plotting
- `Seaborn` - Statistical visualizations
- `WordCloud` - Keyword visualization

### **Database:**
- `SQLite3` - Data warehousing

---

## 📚 Kiến Thức Áp Dụng

### **1. Data Engineering:**
- ✅ Web Scraping at scale
- ✅ Missing Value Imputation (KNN)
- ✅ Outlier Detection (IQR)
- ✅ Data Warehousing (SQL)

### **2. Natural Language Processing:**
- ✅ Text Normalization (Tiếng Việt)
- ✅ TF-IDF Vectorization
- ✅ Feature Selection (Chi-square)
- ✅ Custom Stopwords

### **3. Feature Engineering:**
- ✅ Regular Expression for parsing
- ✅ One-Hot Encoding
- ✅ Frequency Encoding
- ✅ Min-Max Scaling

### **4. Machine Learning:**
- ✅ Ensemble Learning (Voting)
- ✅ Imbalanced Data Handling (SMOTE)
- ✅ Cross-validation & Train-Test Split
- ✅ Hyperparameter Tuning

### **5. Model Evaluation:**
- ✅ Confusion Matrix
- ✅ Precision, Recall, F1-Score
- ✅ Feature Importance Analysis
- ✅ Error Analysis

---

## 💡 Insights & Phát Hiện

### **1. Yếu tố quyết định lương IT tại Việt Nam:**
- 🏆 **Cấp bậc (Level)** và **Kinh nghiệm** là quan trọng nhất (44% importance)
- 🏢 **Quy mô công ty** đóng vai trò lớn (12%)
- 📍 **Địa điểm** (HCM > Hà Nội > Tỉnh) ảnh hưởng ~9%
- 🌐 **Job title tiếng Anh** thường lương cao hơn 15-20%

### **2. Xu hướng thị trường:**
- Middle class chiếm ~60% thị trường
- Senior positions khan hiếm (chỉ ~23% công việc)
- ~30% công ty "giấu lương" (Thỏa thuận)

### **3. Keywords lương cao:**
- "Senior", "Lead", "Manager", "Principal" → +50-100% salary
- "AI", "Data Science", "Cloud", "DevOps" → +30% premium
- "Blockchain", "Machine Learning" → High variance (10tr-100tr+)

---

## 🔮 Hướng Phát Triển

### **Cải tiến ngắn hạn:**
- [ ] Thu thập thêm dữ liệu (target: 5,000+ jobs)
- [ ] Thêm features: Company size, Tech stack requirements
- [ ] Thử nghiệm Deep Learning (BERT for Vietnamese)
- [ ] Build Web API cho inference

### **Cải tiến dài hạn:**
- [ ] Real-time crawling & auto-update model
- [ ] Salary prediction by city/region
- [ ] Recommendation system (công việc phù hợp)
- [ ] Trend analysis dashboard

---

## 📝 Kết Luận

Dự án **IT Salary Classifier** đã thành công trong việc:

✅ Xây dựng pipeline Data Science hoàn chỉnh từ A-Z  
✅ Áp dụng các kỹ thuật tiên tiến (KNN Imputation, SMOTE, Ensemble Learning)  
✅ Đạt độ chính xác ~77% trên tập test  
✅ Giải thích được các yếu tố ảnh hưởng đến lương  
✅ Demo được khả năng dự đoán real-world  

**Điểm nổi bật của dự án:**
- 🎯 **Thực tế**: Dữ liệu thật, bài toán thật
- 🧠 **Kỹ thuật**: Áp dụng SOTA methods
- 📊 **Minh bạch**: Giải thích từng bước, visualize đầy đủ
- 🚀 **Production-ready**: Code sạch, có error handling

---

## 👨‍💻 Tác Giả & Liên Hệ

**Dự án môn học:** Khai Phá Dữ Liệu (Data Mining)  
**Năm thực hiện:** 2025

---

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

---

## 🙏 Lời Cảm Ơn

- **CareerViet.vn** - Nguồn dữ liệu
- **Scikit-learn Community** - Thư viện ML mạnh mẽ
- **Stack Overflow** - Hỗ trợ debug không ngừng nghỉ 😄

---

**⭐ Nếu dự án hữu ích, đừng quên cho một star nhé!**
