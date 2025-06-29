#  Proyek Data Science: Prediksi Attrition Karyawan Perusahaan Edutech

## 🏢 Project Background
Perusahaan Edutech menghadapi permasalahan terkait tingginya angka Attrition (keluar dari perusahaan) pada karyawan. Untuk mengatasi hal ini, proyek ini bertujuan membangun model prediktif yang dapat mengidentifikasi kemungkinan karyawan keluar atau *resign* berdasarkan karakteristik historisnya.

---

## ❗ Problem Statement
Berikut adalah poin-poin masalah yang dihadapi oleh perusahaan Jaya Jaya Maju:

1. Tingginya tingkat *attrition rate* karyawan yang melebihi 10%. 
2. Kurangnya identifikasi faktor penyebab Attrition.
3. Tidak adanya sistem prediksi Attrition. 
4. Keterbatasan *monitoring* dan visualisasi data HR. 

---

## 🎯 Goals
1. Mengidentifikasi faktor-faktor utama yang mempengaruhi tingkat Attrition karyawan.
2. Membangun model *machine learning* yang dapat memprediksi kecenderungan seorang karyawan untuk meninggalkan perusahaan.
3. Membuat *dashboard monitoring* untuk membantu departemen HR dalam memantau faktor-faktor yang mempengaruhi Attrition.

---

## 💡 Solution Approach
Untuk mencapai goals yang telah ditetapkan, berikut adalah solusi yang akan diimplementasikan:

1. Menganalisis dan memvisualisasikan data untuk memahami hubungan antara berbagai variabel dengan tingkat Attrition.
2. Membangun model prediktif untuk memprediksi kecenderungan karyawan untuk meninggalkan perusahaan.
3. Mengevaluasi performa model menggunakan metrik seperti *accuracy*, *precision*, *recall*, dan *F1-score*.
4. Menganalisis *feature importance* untuk mengidentifikasi faktor-faktor utama yang mempengaruhi Attrition.
5. Membuat *dashboard* interaktif menggunakan Metabase untuk *monitoring* faktor-faktor Attrition.

---

## 📊 Data Source
Data yang digunakan dalam proyek adalah [employee_data.csv](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee). Berikut adalah deskripsi variabel-variabel yang terdapat dalam dataset:

- **EmployeeId**: Identitas Karyawan  
- **Attrition**: Apakah karyawan mengundurkan diri? (0 = tidak, 1 = ya)  
- **Age**: Usia karyawan  
- **BusinessTravel**: Komitmen perjalanan dinas untuk pekerjaan  
- **DailyRate**: Gaji harian  
- **Department**: Departemen tempat karyawan bekerja  
- **DistanceFromHome**: Jarak dari rumah ke tempat kerja (dalam km)  
- **Education**: Tingkat pendidikan  
  - 1: Di bawah perguruan tinggi  
  - 2: Perguruan tinggi  
  - 3: Sarjana  
  - 4: Magister  
  - 5: Doktor  
- **EducationField**: Bidang pendidikan  
- **EnvironmentSatisfaction**: Kepuasan terhadap lingkungan kerja  
  - 1: Rendah  
  - 2: Sedang  
  - 3: Tinggi  
  - 4: Sangat Tinggi  
- **Gender**: Jenis kelamin karyawan  
- **HourlyRate**: Gaji per jam  
- **JobInvolvement**: Tingkat keterlibatan kerja  
  - 1: Rendah  
  - 2: Sedang  
  - 3: Tinggi  
  - 4: Sangat Tinggi  
- **JobLevel**: Tingkat jabatan (1 hingga 5)  
- **JobRole**: Peran/jabatan karyawan  
- **JobSatisfaction**: Kepuasan kerja  
  - 1: Rendah  
  - 2: Sedang  
  - 3: Tinggi  
  - 4: Sangat Tinggi  
- **MaritalStatus**: Status pernikahan  
- **MonthlyIncome**: Gaji bulanan  
- **MonthlyRate**: Tarif bulanan  
- **NumCompaniesWorked**: Jumlah perusahaan tempat karyawan pernah bekerja  
- **Over18**: Apakah usia karyawan di atas 18 tahun?  
- **OverTime**: Apakah bekerja lembur?  
- **PercentSalaryHike**: Persentase kenaikan gaji tahun lalu  
- **PerformanceRating**: Penilaian kinerja  
  - 1: Rendah  
  - 2: Baik  
  - 3: Sangat Baik  
  - 4: Luar Biasa  
- **RelationshipSatisfaction**: Kepuasan hubungan kerja  
  - 1: Rendah  
  - 2: Sedang  
  - 3: Tinggi  
  - 4: Sangat Tinggi  
- **StandardHours**: Jam kerja standar  
- **StockOptionLevel**: Tingkat opsi saham  
- **TotalWorkingYears**: Total tahun pengalaman kerja  
- **TrainingTimesLastYear**: Jumlah pelatihan yang diikuti tahun lalu  
- **WorkLifeBalance**: Keseimbangan kerja dan kehidupan pribadi  
  - 1: Rendah  
  - 2: Baik  
  - 3: Sangat Baik  
  - 4: Luar Biasa  
- **YearsAtCompany**: Jumlah tahun bekerja di perusahaan  
- **YearsInCurrentRole**: Jumlah tahun dalam peran saat ini  
- **YearsSinceLastPromotion**: Jumlah tahun sejak promosi terakhir  
- **YearsWithCurrManager**: Jumlah tahun bersama manajer saat ini

---

## 📦 Project Structure

```
submission/
├── notebook.ipynb
├── notebook.py
├── requirements.txt
├── idhakt - dashboard.png
├── README.md
├── metabase.db.mv.db
├── Random Forest_model.pkl
└── prediction.ipnyb
```

---

## 🔧 System Requirements

| Komponen         | Versi                |
|------------------|----------------------|
| Python           | 3.9.2                |
| Docker Desktop   | 28.0.4               |
| MySQL            | 8.0.40               |
| DBeaver          | 25.0.4               |
| OS               | Windows 64-bit       |

---

## 🧱 Dependencies

### ⚙️ Setup Environment
##### 🐍 Anaconda
```
bashconda create --name jaya-attrition python=3.9
conda activate jaya-attrition
pip install -r requirements.txt
```

##### 📟 Shell/Terminal
```
bashpip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```

##### ☁️ Virtual Environment
```
bashpython -m venv jaya-env
# Windows
jaya-env\Scripts\activate
# MacOS/Linux
source jaya-env/bin/activate
pip install -r requirements.txt
```


### 🐳 Dockerfile

```Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libmysqlclient-dev \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 3000

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=3000", "--no-browser", "--allow-root"]
```

### 🚀 Running the Project with Docker

```bash
docker pull metabase/metabase:v0.46.4
docker run -p 3000:3000 --name metabase metabase/metabase
```

Buka di browser: `http://localhost:3000/setup`

### 🔐 Log In Metabase

- **Email**     : `root@mail.com`
- **Password**  : `root123`

### ▶️ Running the Prediction File

```
python prediction.ipnyb
```

---


## 🔍 *Insight* Analysis

### **🔢 Univariate Analysis Data Numerik**
![image](https://github.com/user-attachments/assets/49b23c54-b369-4c92-ba0c-d647ead3b92a)

Berikut *insight* dari visualisasi di atas.

| No | Fitur                       | Distribusi                                                                 | Boxplot Insight                                                                 |
|----|-----------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| 1  | EmployeId                    | Merata, jumlah konsisten di seluruh rentang                                | Tidak ada outlier                                                    |
| 2  | Age                         | Distribusi normal, puncak sekitar usia 35                                   | Sebaran antara 30–45 tahun, tidak ada outlier                                   |
| 3  | Attrition                   | Kelas tidak seimbang, dominan “No”                                          | Sedikit outlier di kelas “Yes”                                                  |
| 4  | DailyRate                   | Relatif seragam                                                            | Sebaran luas, tidak ada outlier                                      |
| 5  | DistanceFromHome           | Mayoritas <5 km, menurun tajam setelahnya                                   | Tidak ada outlier dan sebaran data cenderung miring ke kanan (right-skewed) |
| 6  | Education                  | Dominan di level 3                                                         | Tidak ada outlier, data merata                                                  |
| 7  | EnvironmentSatisfaction     | Tinggi pada level 1 dan 4                                                  | Tidak ada outlier dan sebaran data  cenderung miring ke kiri (left-skewed) atau negatif                   |
| 8  | HourlyRate                 | Cenderung seragam                                                          | Tidak ada outlier, sebaran merata                                               |
| 9  | JobInvolvement              | Mayoritas di level 3                                                       | Tidak ada outlier, sebaran merata                                                               |
| 10 | JobLevel                   | Dominan di level 1 dan 2                                                   | Tidak ada outlier dan sebaran data cenderung miring ke kanan (right-skewed) atau positif                                             |
| 11 | JobSatisfaction             | Tinggi di level 3 dan 4                                                    | Tidak ada outlier dan sebaran data  cenderung miring ke kiri (left-skewed) atau negatif                                                               |
| 12 | MonthlyIncome              | Mayoritas berpenghasilan rendah                            | Banyak outlier di penghasilan tinggi dan sebaran  data cenderung miring ke kanan (right-skewed) atau positif                                            |
| 13 | NumCompaniesWorked         | Dominan di 0–2 perusahaan                                                  | Terdapat outlier yang tidak signifikan dan sebaran  data cenderung miring ke kanan (right-skewed) atau positif                   |
| 14 | PercentSalaryHike          | Positif skewed                                                             | Tidak ada outlier dan sebaran  data cenderung miring ke kanan (right-skewed) atau positif                                                             |
| 15 | PerformanceRating           | Hampir semua di level 3                                                    | Sedikit outlier di level 4                                                      |
| 16 | RelationshipSatisfaction    | Mayoritas berada di 3 dan 4                                                  | Tidak ada outlier dan sebaran data  cenderung miring ke kiri (left-skewed) atau negatif                                                                |
| 17 | StockOptionLevel            | Dominan level 0 dan 1                                                           | Sedikit outlier pada level 3 dan  sebaran data cenderung miring ke kanan (right-skewed) atau positif                                |
| 18 | TotalWorkingYears          | Positif skewed, mayoritas <10 tahun                                        | Banyak outlier >30 tahun dan sebaran data cenderung miring ke kanan (right-skewed) atau positif                         |
| 19 | TrainingTimesLastYear      | Mayoritas 2–3 kali                                                         | Outlier pada pelatihan >5 kali  dan 0 kali                                                |
| 20 | YearsAtCompany             | Positif skewed, mayoritas <5 tahun                                         | Terdapat banyak outlier pada masa kerja >15 tahun dan  sebaran data cenderung miring ke kanan (right-skewed) atau positif                                              |
| 21 | YearsInCurrentRole         | Mayoritas <5 tahun                                                         | Terdapat outlier pada >15 tahun di peran kerja dan sebaran data cenderung miring ke kanan (right-skewed) atau positif                                                |
| 22 | YearsSinceLastPromotion    | Mayoritas 0 tahun                                                          | Outlier pada >8 tahun tanpa promosi dan sebaran data cenderung miring ke kanan (right-skewed) atau positif                                           |
| 23 | YearsWithCurrManager       | Mayoritas <5 tahun                                                         | Outlier pada hubungan >15 tahun dengan manajer dan  sebaran data cenderung miring ke kanan (right-skewed) atau positif                                 |
| 24  | MonthlyRate                   | Relatif seragam                                                            | Sebaran luas, tidak ada outlier                                      |
| 25  | WorkLifeBalance                   | mayoritas pada 3                                                            | Sebaran luas, tidak ada outlier                                      |


### **🔢 Univariate Analysis Data Kategorial**
![image](https://github.com/user-attachments/assets/84607214-0b6a-474f-a74e-3cb49d20d00d)

Berikut *insight* dari visualisasi di atas.
| Variabel             | Kategori Dominan         | *Insight*|
|----------------------|---------------------------|---------------------|
| **BusinessTravel**   | Travel_Rarely             | Mayoritas karyawan jarang melakukan perjalanan bisnis, menunjukkan mobilitas moderat dalam pekerjaan mereka. |
| **Department**       | Research & Development    | Sebagian besar karyawan bekerja di departemen R&D, menandakan fokus utama perusahaan pada inovasi dan pengembangan produk. |
| **EducationField**   | Life Sciences             | Latar belakang pendidikan paling umum adalah Life Sciences, yang sesuai dengan dominasi peran teknis dan ilmiah. |
| **Gender**           | Male                      | Karyawan laki-laki lebih banyak daripada perempuan dalam perusahaan ini. |
| **JobRole**          | Sales Executive           | Peran paling umum adalah Sales Executive, menunjukkan pentingnya divisi penjualan dalam struktur organisasi. |
| **MaritalStatus**    | Married                   | Mayoritas karyawan telah menikah. |
| **OverTime**         | No                        | Lebih banyak karyawan tidak lembur, yang bisa menandakan beban kerja sedang atau kebijakan perusahaan terhadap *work-life balance*. |


### **📊 Visualisasi Multivariate dengan Bar Chart**
![image](https://github.com/user-attachments/assets/5b82f5b6-354a-45dc-8eee-76ba52b23861)

Berikut *insight* dari visualisasi bar di atas.
| **Variabel**           | **Kategori Signifikan**                            | **Temuan**                                                                 | ***Insight***                                                                                          |
|------------------------|----------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **BusinessTravel**     | Travel_Frequently, Travel_Rarely                   | Travel_Frequently memiliki Attrition tinggi.                              | Karyawan yang sering bepergian berisiko lebih tinggi mengalami kelelahan kerja.                      |
| **Department**         | Sales, R&D                              | R&D memiliki Attrition tinggi.              | Peran R&D mendapat tekanan besar untuk menghasilkan inovasi dalam waktu singkat.   |
| **EducationField**     | Life Sciences                        | Life Sciences memiliki Attrition yang tinggi.           | Latar belakang pendidikan mungkin tidak sesuai ekspektasi karir.             |
| **Gender**             | Male                              | Male memiliki Attrition yang tinggi. gender.                                | Salah satu alasan potensial mengapa laki-laki (Male) memiliki tingkat Attrition yang tinggi adalah kurangnya keseimbangan antara pekerjaan dan kehidupan pribadi. |
| **JobRole**            | Sales Executive, Labotary Technician, dan Research Scientist   | Job role seperti Sales Executive, Labotary Technician, dan Research Scientist memiliki Attrition tinggi. | Role dengan tekanan tinggi.                           |
| **MaritalStatus**      | Single                                             | Single punya tingkat Attrition lebih tinggi daripada Married.             | Karyawan *single* lebih *mobile* dan terbuka terhadap peluang lain.                                      |
| **OverTime**           | Yes                                                | Tingkat Attrition antara karyawan yang lembur dan yang tidak lembur hampir sama.                       | Lembur berlebihan memicu kelelahan dan niat *resign* yang lebih besar.                                |

### **📊 Visualisasi Multivariate dengan Boxplot**
![image](https://github.com/user-attachments/assets/a1dfcc51-bd4d-45f9-8802-3299bed22302)

Berikut *insight* dari visualisasi bar di atas.

| **Variabel**                 | **Insight**                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------------|
| **EmployeeID**              | Tidak menunjukkan perbedaan signifikan antara yang *resign* dan tidak (hanya ID urutan).           |
| **Age**                     | Karyawan yang *resign* cenderung lebih muda.                                                       |
| **DailyRate**               | Tidak ada perbedaan yang mencolok antara kelompok *resign* dan tidak.                              |
| **DistanceFromHome**        | Semakin jauh jarak dari rumah, semakin tinggi kecenderungan untuk *resign.*                        |
| **Education**               | Tingkat pendidikan tidak terlalu memengaruhi kecenderungan *resign.*                              |
| **EnvironmentSatisfaction** | Median kepuasan lingkungan kerja lebih rendah pada karyawan yang *resign.*                         |
| **HourlyRate**              | Tidak ada pola signifikan antara HourlyRate dan Attrition.                                      |
| **JobInvolvement**          | Tingkat keterlibatan kerja serupa antara yang *resign* dan tidak.                                  |
| **JobLevel**                | Mayoritas karyawan yang *resign* berada di level jabatan lebih rendah.                            |
| **JobSatisfaction**         | Karyawan yang *resign* memiliki tingkat kepuasan kerja yang rendah.                      |
| **MonthlyIncome**           | Karyawan yang *resign* memiliki pendapatan bulanan lebih rendah.                                   |
| **MonthlyRate**             | Tidak tampak perbedaan signifikan antara yang *resign* dan tidak.                                  |
| **NumCompaniesWorked**      | Karyawan yang *resign* cenderung telah bekerja di lebih banyak perusahaan sebelumnya.              |
| **PercentSalaryHike**       | Tidak ada perbedaan yang jelas terhadap persentase kenaikan gaji.                               |
| **PerformanceRating**       | Semua karyawan memiliki rating performa yang hampir seragam.                                     |
| **RelationshipSatisfaction**| Distribusi dan median sedikit lebih rendah untuk yang *resign.*                                   |
| **StockOptionLevel**        | Tidak menunjukkan pola yang signifikan terhadap attrition.                                       |
| **TotalWorkingYears**       | Karyawan yang *resign* memiliki pengalaman kerja lebih sedikit.                                   |
| **TrainingTimesLastYear**   | Jumlah pelatihan terakhir tidak terlalu berpengaruh.                             |
| **WorkLifeBalance**         | WorkLifeBalance tidak terlalu memengaruhi keputusan *resign* secara jelas.                |
| **YearsAtCompany**          | Karyawan *resign* umumnya baru bekerja sebentar (median rendah).                                  |
| **YearsInCurrentRole**      | Umumnya memiliki masa kerja lebih pendek di peran saat ini.                                     |
| **YearsSinceLastPromotion** | Karyawan yang *resign* tidak lama setelah promosi terakhir.                                       |
| **YearsWithCurrManager**    | Masa kerja dengan manajer saat ini lebih pendek di kelompok *resign.*                             |



### **📊 Distribusi Attrition**
![image](https://github.com/user-attachments/assets/55f25f04-c948-4741-ae89-b75200cbda42)

Visualisasi di atas menunjukkan distribusi Attrition (pengunduran diri) karyawan dalam dataset, dengan:

1. Kategori 0 (Tidak mengundurkan diri)
Mayoritas karyawan dalam dataset berada dalam kategori ini. Secara visual terlihat bahwa lebih dari 850 karyawan masih aktif bekerja di perusahaan. Hal ini menunjukkan bahwa sebagian besar tenaga kerja tetap bertahan dan tidak meninggalkan perusahaan.

2. Kategori 1 (Mengundurkan diri)
Hanya sebagian kecil karyawan yang termasuk dalam kategori ini, yaitu sekitar 150 hingga 200 orang. Jumlah ini jauh lebih rendah dibandingkan karyawan yang bertahan.

*Insight* pada visualisasi diagram bar di atas, yaitu mayoritas karyawan tidak mengundurkan diri, karyawan yang mengundurkan diri (1) hanya sekitar 150–200 orang, dan tingkat Attrition secara kasar berkisar antara 15%–20%.


### **🔢 Matriks Korelasi Antar Variabel Numerik**
![image](https://github.com/user-attachments/assets/5fb4cfa1-0632-48c6-9220-e0149d92c5c2)

Visualisasi di atas merupakan matriks korelasi antar variabel numerik. Korelasi Attrition dengan semua variabel numerik berada di bawah 0.08, yang berarti tidak ada variabel dengan korelasi kuat. Berikut pasangan fitur yang memiliki korelasi positif tinggi.
| Pasangan Variabel                  | Nilai Korelasi | *Insight*                                                                 |
|-----------------------------------|----------------|------------------------------------------------------------------------|
| **MonthlyIncome – JobLevel**      | 0.95           | Gaji sangat berkorelasi dengan level jabatan. Semakin tinggi level, semakin besar penghasilan. |
| **TotalWorkingYears – MonthlyIncome** | 0.78      | Semakin lama pengalaman kerja, semakin tinggi penghasilan.            |
| **TotalWorkingYears – JobLevel**  | 0.79           | Pengalaman kerja juga berkaitan dengan level jabatan.                 |
| **YearsAtCompany – YearsWithCurrManager** | 0.76   | Masa kerja di perusahaan sangat berkaitan dengan masa kerja dengan manajer saat ini. |
| **YearsAtCompany – YearsInCurrentRole** | 0.76   | Lama bekerja di perusahaan berkorelasi erat dengan lama di posisi sekarang. |



### **🔢 Matriks Korelasi Seluruh Fitur**
![image](https://github.com/user-attachments/assets/c4587672-21ee-445f-affc-c81d1da117f7)

Visualisasi di atas merupakan matriks korelasi antara semua variabel, baik variabel numerik maupun kategorial yang sudah dilakukan *one hot encoding*. Terdapat beberapa *insight* pada visualisasi di atas, yaitu pasangan fitur `MonthlyIncome` dan `JobLevel`	merupakan pasangan fitur yang memiliki korelasi positif paling tinggi, sebesar 0.95. Korelasi `Attrition` dengan sebagian besar variabel bersifat lemah, yaitu di bawah 0.26.



### **📊 Visualisasi 15 Fitur yang Paling Berkorelasi dengan Attrition**
![image](https://github.com/user-attachments/assets/a400d15f-6602-4612-b402-4b73f0afc164)

Grafik di atas menunjukkan 15 fitur yang memiliki korelasi paling tinggi (positif) terhadap variabel target Attrition. Berikut *insight* dari grafik di atas.

| No | Fitur                             | Korelasi | Interpretasi                                                                 |
|----|-----------------------------------|----------|------------------------------------------------------------------------------|
| 1  | **OverTime_Yes**                  | 0.26     | Karyawan yang sering lembur cenderung lebih tinggi kemungkinannya keluar.   |
| 2  | **MaritalStatus_Single**         | 0.18     | Karyawan lajang lebih rentan keluar daripada yang sudah menikah.            |
| 3  | **JobRole_Sales Representative** | 0.17     | Posisi ini menunjukkan tingkat keluar yang relatif tinggi.                  |
| 4  | **JobRole_Laboratory Technician**| 0.11     | Laboratorium technician juga rentan terhadap Attrition.                     |
| 5  | **BusinessTravel_Travel_Frequently** | 0.10  | Karyawan yang sering dinas keluar kota lebih rentan mengundurkan diri.      |
| 6  | **DistanceFromHome**             | 0.08     | Semakin jauh rumah dari kantor, semakin besar kemungkinan *resign*.           |
| 7  | **EducationField_Technical Degree** | 0.08   | Latar pendidikan teknikal sedikit lebih rentan terhadap Attrition.          |
| 8  | **Department_Sales**             | 0.07     | Departemen Sales menunjukkan tingkat pengunduran diri yang cukup tinggi.    |
| 9  | **EducationField_Marketing**     | 0.04     | Latar belakang marketing juga punya sedikit kontribusi terhadap Attrition.  |
| 10 | **NumCompaniesWorked**           | 0.04     | Karyawan yang pernah bekerja di banyak perusahaan cenderung keluar lagi.    |
| 11 | **MonthlyRate**                  | 0.02     | Pengaruh kecil dari besar kecilnya gaji bulanan terhadap keputusan keluar.  |
| 12 | **Gender_Male**                  | 0.02     | Laki-laki sedikit lebih mungkin keluar, tapi pengaruhnya sangat kecil.      |
| 13 | **JobRole_Human Resources**      | 0.01     | Posisi ini hanya menunjukkan korelasi kecil terhadap Attrition.             |
| 14 | **JobRole_Research Scientist**   | 0.01     | Sama seperti HR, peran ini relatif stabil.                                  |
| 15 | **PerformanceRating**            | 0.01     | Penilaian kinerja ternyata tidak banyak berpengaruh terhadap Attrition.     |

Berdasarkan pemaparan di atas, dapat diambil kesimpulan yaitu sebagai berikut.
- Fitur yang paling kuat korelasinya dengan Attrition adalah **OverTime_Yes**.
- Korelasi masih dalam kategori **lemah** karena nilai yang paling tinggi hanya sebesar 0.26.
- Fitur-fitur seperti OverTime, JobRole, MaritalStatus, BusinessTravel, dan DistanceFromHome dapat dijadikan **indikator awal risiko Attrition**.

---

## 🤖 Modeling
Berikut beberapa model machine learning telah diuji untuk memprediksi Attrition karyawan:
| Model               | Accuracy  | Precision | Recall   | F1 Score | ROC AUC  | Training Time (s) | Rata-rata Skor |
|---------------------|-----------|-----------|----------|----------|----------|-------------------|----------------|
| Random Forest       | 0.931818  | 0.946429  | 0.913793 | 0.929825 | 0.971232 | 0.535974          | 0.930466       |
| XGBoost             | 0.917614  | 0.944785  | 0.885057 | 0.913947 | 0.962192 | 2.975441          | 0.915351       |
| Gradient Boosting   | 0.914773  | 0.933735  | 0.890805 | 0.911765 | 0.963063 | 0.834926          | 0.912769       |
| Logistic Regression | 0.909091  | 0.943750  | 0.867816 | 0.904192 | 0.962095 | 0.038560          | 0.906212       |
| KNN                 | 0.903409  | 0.893258  | 0.913793 | 0.903409 | 0.973185 | 0.074894          | 0.903467       |
| SVM                 | 0.906250  | 0.954839  | 0.850575 | 0.899696 | 0.964839 | 0.631115          | 0.902840       |
| Decision Tree       | 0.826705  | 0.815642  | 0.839080 | 0.827195 | 0.826844 | 0.029770          | 0.827156       |


Berdasarkan data performa model yang diberikan, Random Forest adalah model terbaik untuk memprediksi Attrition karyawan. Model Random Forest  cocok untuk kasus Attrition karena beberapa alasan, yaitu sebagai berikut.
1. Random Forest mampu menangani data campuran—baik numerik (misalnya, `MonthlyIncome`) maupun kategorikal (misalnya, `OverTime`, `Department`, `JobRole`) tanpa perlu transformasi kompleks.
2. Dengan skor evaluasi tinggi seperti F1 Score **0.9298**, Random Forest menunjukkan performa yang sangat baik dalam mengklasifikasi antara karyawan yang tetap dan yang keluar.
3. Random Forest adalah gabungan dari banyak pohon keputusan (*decision trees*) yang dibangun dari subset data dan fitur berbeda, sehingga hasil prediksi lebih stabil dan tidak mudah overfit terhadap data training.
4. Model ini menyediakan metrik *feature importance*, yang memungkinkan perusahaan mengetahui fitur mana yang paling berpengaruh terhadap keputusan resign, seperti `OverTime`, `JobInvolvement`, `StockOptionLevel`, `EnvironmentSatisfaction`, dan `MonthlyIncome`.
5. Model ini cukup efisien dan dapat di-*parallelize*, sehingga bisa digunakan pada dataset berskala besar atau diproses di lingkungan produksi dengan resource komputasi terbatas.
6. Random Forest relatif tidak sensitif terhadap data yang memiliki *missing value* atau nilai ekstrim (*outlier*), sehingga cocok untuk data HR yang sering memiliki kekurangan atau ketidaksempurnaan.

---

## 📊 Model Evaluation

### 📈 Visualisasi ROC Curve (Receiver Operating Characteristic Curve)
![image](https://github.com/user-attachments/assets/a05a17b3-df1f-41ff-a2a8-5eab1900626a)

Berdasarkan visualisasi di atas, model Random Forest merupakan model terbaik untuk kasus prediksi Attrition karyawan, berdasarkan kurva ROC yang sangat baik dan nilai AUC = 0.9757, yang menunjukkan bahwa model ini dapat mengklasifikasikan dengan sangat baik antara karyawan yang akan keluar dan yang akan tetap tinggal. Kurva ROC model Random Forest berada jauh di atas garis diagonal (*baseline*), yang mengindikasikan bahwa model memiliki kemampuan klasifikasi yang sangat baik dalam membedakan antara kelas positif dan kelas negatif. ROC Curve menunjukkan bahwa untuk mayoritas titik *threshold*, model menjaga tingkat *false positive* yang rendah sambil tetap mempertahankan *true positive rate* yang tinggi. Dengan TPR yang tinggi dan FPR yang rendah, Random Forest cocok untuk diterapkan pada kasus di mana:
1. Kelas minoritas (seperti karyawan keluar/Attrition) penting untuk dikenali.
2. Kesalahan klasifikasi (*false negative* atau *false positive*) memiliki konsekuensi besar.

### 🔢 Visualisasi Confusion Matrix untuk Model Random Forest
![image](https://github.com/user-attachments/assets/e96d0441-0bf0-416c-8679-6d37e6aa612e)

Berdasarkan visualisasi di atas, dapat diambil kesimpulan yaitu sebagai berikut.
1. Model Random Forest memiliki performa klasifikasi yang sangat baik, dengan jumlah prediksi benar (TP + TN) yang sangat tinggi.
2. Kesalahan prediksi (FP dan FN) tergolong kecil, menandakan model stabil dan tidak bias terhadap salah satu kelas.
3. Cocok untuk digunakan pada kasus nyata dalam sistem HR untuk mengantisipasi potensi pengunduran diri karyawan secara proaktif.

### 📈 Visualisasi Feature Importance dari Model Random Forest
![image](https://github.com/user-attachments/assets/3e7aa544-8622-4ab2-a40f-304b6452831c)

Grafik di atas menunjukkan pentingnya setiap fitur dalam model Random Forest untuk memprediksi **attrition** (berhentinya karyawan). Nilai pada sumbu horizontal menunjukkan tingkat kontribusi relatif (koefisien korelasi) dari masing-masing fitur terhadap prediksi model.

🔼 **Fitur-Fitur Penting Teratas**
1. `OverTime_No` (0.11) 
   - Fitur ini merupakan fitur yang paling berpengaruh. Karyawan yang **tidak lembur** cenderung **lebih kecil** kemungkinannya untuk keluar dari perusahaan.
   
2. `JobInvolvement` (0.05)  
   - Tingkat keterlibatan kerja menjadi faktor utama. Karyawan dengan **keterlibatan tinggi** menunjukkan kemungkinan bertahan lebih besar.
   
3. `StockOptionLevel` (0.05)  
   - Karyawan dengan **opsi saham** memiliki motivasi tambahan untuk tetap tinggal.

4. `EnvironmentSatisfaction`, `MonthlyIncome`, `TotalWorkingYears` (0.04)  
   - **Kepuasan lingkungan kerja**, **penghasilan bulanan**, dan **lama bekerja** juga berkontribusi penting terhadap keputusan untuk tetap atau keluar.

5. `JobSatisfaction`, `Age`, `JobLevel`, `HourlyRate`, `DailyRate`, `MaritalStatus_Married`, `YearsWithCurrManager` dan `YearsInCurrentRole` memiliki kontribusi sedang (sekitar 0.03), menunjukkan bahwa faktor tersebut memengaruhi keputusan karyawan.


🔽 **Fitur dengan Pengaruh Rendah**
- Banyak fitur yang memiliki nilai mendekati **0.00**, seperti:
  - `JobRole_Sales Executive`
  - `BusinessTravel_Travel_Frequently`
  - `PerformanceRating`
  - `EducationField_Marketing`
  - `EducationField_Technical Degree`
  - `JobRole_Sales Representative`
  - `EducationField_Other`
  - `JobRole_Healthcare Representative`
  - `Department_Human Resources`
  - `JobRole_Research Director`
  - `JobRole_Manager`
  - `JobRole_Human Resources`
  - `EducationField_Human Resources`

  Hal ini menunjukkan bahwa **peran spesifik** atau **bidang pendidikan** tidak terlalu menentukan apakah seseorang akan keluar dari perusahaan.


---

## 🏁 Conclusion

Berdasarkan hasil analisis dan pemodelan yang telah dilakukan pada dataset karyawan perusahaan Jaya Jaya Maju, berikut adalah kesimpulan dari *problem statement* yang telah diidentifikasi:

- **Tingginya Tingkat Attrition Rate Karyawan yang Melebihi 10%**

  Tingkat Attrition yang melebihi 10% menjadi perhatian serius bagi perusahaan. Model Random Forest yang dikembangkan telah berhasil mengidentifikasi faktor-faktor kunci yang berkontribusi terhadap tingginya angka Attrition ini. Dengan memahami faktor-faktor ini, perusahaan dapat mengambil langkah-langkah proaktif untuk menekan angka attrition. Model mengidentifikasi bahwa lima faktor teratas yang berkontribusi terhadap Attrition tinggi meliputi:

    - OverTime (lembur berlebihan)
    - Keterlibatan dalam pekerjaan (Job Involvement)
    - Level opsi saham (Stock Option Level)
    - Kepuasan terhadap lingkungan kerja (Environment Satisfaction)
    - Pendapatan bulanan (Monthly Income)

- **Faktor-faktor yang Mempengaruhi Tingginya Tingkat Attrition**

  Model Random Forest berhasil mengidentifikasi faktor-faktor penyebab Attrition secara kuantitatif. Berikut faktor teridentifikasi sebagai faktor yang mempengaruhi tingkat Attrition berdasarkan *feature importance*:

  - OverTime_No (0.11) 
  - JobInvolvement (0.05) 
  - StockOptionLevel (0.05) 
  - EnvironmentSatisfaction (0.04) 
  - MonthlyIncome (0.04) 
  - TotalWorkingYears (0.04) 
  - MonthlyRate (0.04) 

- **Cara Memprediksi Kecenderungan Karyawan untuk Meninggalkan Perusahaan**

  Dengan dikembangkannya model Random Forest yang memiliki F1 Score 0.9298, perusahaan kini memiliki sistem prediksi Attrition. Model ini dapat digunakan untuk mengidentifikasi karyawan yang berpotensi tinggi untuk Attrition berdasarkan karakteristik dan data.


- **Cara Memvisualisasikan Data untuk Wawasan yang Lebih Baik**

  Cara memvisualisasikan data secara efektif dan memberikan wawasan yang *actionable* kepada departemen HR yaitu dengan membuat *dashboard monitoring* Attrition agar dapat mengidentifikasi departemen atau kelompok karyawan yang berisiko tinggi, memahami faktor pendorong utama Attrition, merancang intervensi yang tepat sasaran, dan mengukur efektivitas program retensi karyawan dari waktu ke waktu. Dashboard ini dapat dibuat menggunakan tools seperti Metabase, Streamlit, Tableau, Power BI, atau bahkan kombinasi dari Matplotlib dan Seaborn di Python yang kemudian diintegrasikan ke dalam aplikasi web sederhana.

---

## 💼 Action Items

Berikut adalah rekomendasi *action items* untuk HR dan manajemen berdasarkan lima fitur teratas yang mempengaruhi Attrition:

1. OverTime
    - Kaji ulang kebijakan lembur.
    - Distribusi beban kerja yang merata.
    - Implementasi *work life balance*.

2. Job Involvement
    - Implementasikan program mentoring dan pemberdayaan karyawan.
    - Adakan aktivitas *team building* secara berkala.
    - Berikan proyek yang lebih menantang dan bermakna sesuai minat karyawan.
    - Lakukan survei keterlibatan secara berkala.
    - Berikan pengakuan dan penghargaan untuk kontribusi karyawan.

3. Stock Option
    - Evaluasi kebijakan opsi saham.
    - Tawarkan program kepemilikan saham kepada karyawan potensial.
    - Jadikan *stock option* sebagai insentif jangka panjang untuk retensi.

4. Environment Satisfaction
    - Survei kepuasan lingkungan kerja.
    - Tingkatkan fasilitas dan kenyamanan kerja.
    - Ciptakan budaya kerja positif dan kolaboratif.

5. Monthly Income
    - Tinjau dan evaluasi struktur gaji dan tunjangan.
    - Implementasi program apresiasi dan penghargaan non-finansial untuk mengakui kinerja baik karyawan.


Dengan mengimplementasikan *action items* ini secara sistematis dan terukur, Perusahaan Jaya Jaya Maju dapat mengurangi tingkat Attrition secara signifikan.
