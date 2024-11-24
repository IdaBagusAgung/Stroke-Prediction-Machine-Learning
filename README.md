# Proyek Pertama Machine Learning Terapan | Stroke Prediction using Machine Learning and Deep Learning

###### Disusun oleh : Ida Bagus Agung Bajerapany

Proyek ini membangun model *machine learning* dan *Deep Learning* yang dapat memprediksi stroke berdasarkan indikator stroke

## 1. Project Domain

Stroke merupakan salah satu penyebab utama kematian di seluruh dunia, menempati posisi kedua sebagai penyebab kematian global, yang menunjukkan betapa seriusnya dampak penyakit ini terhadap kesehatan masyarakat di berbagai negara.【1】. Di Indonesia, kondisi ini menyumbang sekitar 7,9% dari seluruh kematian yang terjadi, yang menjadikannya salah satu ancaman kesehatan yang paling signifikan dan permasalahan yang perlu mendapatkan perhatian khusus dari para penyedia layanan kesehatan dan pemerintah【1】. Lebih jauh lagi, sebuah laporan dari Institute for Health Metrics and Evaluation (IHME) pada tahun 2019 mengungkapkan bahwa stroke menyumbang 19,42% dari seluruh penyebab kematian di Indonesia, angka yang mencerminkan beban yang berat dari penyakit ini dan perlunya upaya lebih lanjut dalam pencegahan dan penanganannya agar dapat mengurangi angka kematian yang disebabkan oleh stroke di masa depan【2】.

Faktor risiko utama yang berkontribusi pada terjadinya stroke, seperti hipertensi, diabetes, obesitas, dan pola hidup yang tidak sehat, sebenarnya dapat diidentifikasi lebih awal dan secara sistematis melalui analisis data kesehatan yang komprehensif dan berkelanjutan. Melalui pengumpulan dan evaluasi data mengenai kebiasaan hidup, riwayat kesehatan individu, serta faktor lingkungan, para profesional kesehatan dapat mengidentifikasi individu yang berada dalam kategori risiko tinggi sebelum mereka mengalami kondisi yang lebih serius. Sayangnya, sering kali terjadi keterlambatan dalam proses deteksi ini, yang dapat berujung pada konsekuensi yang sangat serius, termasuk kematian mendadak atau, dalam banyak kasus, disabilitas permanen yang mengubah kualitas hidup seseorang secara drastis.

Pentingnya pendekatan preventif terhadap masalah ini semakin diperkuat oleh berbagai temuan penelitian yang menunjukkan bahwa sekitar 80% kasus stroke dapat dicegah jika faktor risiko yang ada dikelola dengan baik. Ini mencakup menerapkan pola makan yang sehat dan seimbang, berolahraga secara rutin untuk menjaga kebugaran fisik, serta secara aktif mengendalikan tekanan darah untuk mencegah fluktuasi yang berbahaya. Dengan meningkatkan kesadaran dan menyediakan sumber daya untuk mengelola faktor-faktor ini, kita dapat secara signifikan mengurangi angka kejadian stroke dan meningkatkan kualitas hidup masyarakat secara keseluruhan. Dalam konteks ini, peran edukasi kesehatan dan kampanye pencegahan menjadi sangat krusial dalam membantu masyarakat memahami pentingnya menjaga kesehatan untuk mencegah stroke dan penyakit terkait lainnya.

### Bagaimana Masalah tersebut Diselesaikan?

#### 1. Pemanfaatan Teknologi Prediksi
Dengan algoritma pembelajaran mesin seperti Neural Network dan Support Vector Machine (SVM), prediksi risiko stroke dapat dilakukan secara lebih cepat dan akurat. Penelitian menunjukkan bahwa metode Neural Network dengan seleksi fitur dapat mencapai akurasi hingga 88,75%, sementara SVM dengan teknik SMOTE menghasilkan akurasi hingga 85,45%【1】【2】.

#### 2. Deteksi Dini Melalui Analisis Data
Dengan memanfaatkan dataset seperti Stroke Prediction Dataset dari Kaggle, faktor-faktor utama seperti umur, riwayat hipertensi, BMI, dan kadar glukosa darah dapat diidentifikasi sebagai indikator risiko. Model prediksi ini memungkinkan deteksi dini sehingga individu dengan risiko tinggi dapat segera memperoleh intervensi【1】【2】.

#### 3. Edukasi dan Kesadaran Masyarakat
Hasil analisis dapat digunakan untuk mengedukasi masyarakat tentang pentingnya gaya hidup sehat dan pengelolaan risiko. Misalnya, mengurangi konsumsi makanan berlemak tinggi, rutin berolahraga, dan berhenti merokok【2】.

#### 4. Efisiensi Sistem Kesehatan
Teknologi prediksi berbasis data membantu dokter dalam pengambilan keputusan, memungkinkan alokasi sumber daya kesehatan yang lebih efisien dengan fokus pada pasien berisiko tinggi.

#### 5. Dampak yang Diharapkan
Dengan pendekatan ini, angka kejadian stroke dapat ditekan secara signifikan, mengurangi beban ekonomi pada keluarga dan sistem kesehatan. Selain itu, model ini dapat mendukung pengembangan aplikasi prediksi berbasis web atau mobile untuk memudahkan akses masyarakat terhadap deteksi risiko stroke【1】【2】.


### State Of The Art Penelitian Sebelumnya

1. Penelitian yang dilakukan oleh Werdiningsih et al. (2023) berhasil menggabungkan metode Decision Tree untuk proses seleksi fitur dan Neural Network untuk klasifikasi data pada sebuah dataset yang diperoleh dari Kaggle, yang merupakan salah satu platform terkenal untuk analisis data dan kompetisi mesin belajar. Dalam penelitian ini, berbagai fitur yang relevan dipilih untuk dianalisis, antara lain umur, riwayat hipertensi, status pernikahan, Body Mass Index (BMI), dan kadar glukosa darah rata-rata, yang semuanya dianggap berpengaruh signifikan terhadap risiko terkena stroke. Hasil dari pendekatan yang digunakan menunjukkan bahwa penerapan seleksi fitur dapat meningkatkan akurasi prediksi model hingga mencapai 88,75% ketika menggunakan Neural Network, yang menunjukkan performa yang lebih baik dibandingkan dengan akurasi 81,25% yang dicapai oleh model Decision Tree tanpa melalui proses seleksi fitur tersebut. Temuan ini secara jelas mengindikasikan bahwa pemilihan fitur yang relevan dan tepat tidak hanya berkontribusi pada peningkatan akurasi prediksi, tetapi juga mampu mengoptimalkan kinerja model pembelajaran mesin secara keseluruhan, terutama dalam konteks prediksi risiko stroke, yang merupakan langkah penting dalam upaya pencegahan dan pengelolaan kondisi kesehatan yang serius ini【1】.

2. Penelitian yang dilakukan oleh Wulandari et al. (2024) menerapkan algoritma Support Vector Machine (SVM) yang dikombinasikan dengan teknik SMOTE (Synthetic Minority Oversampling Technique) untuk secara efektif menangani masalah ketidakseimbangan yang sering terjadi dalam data, terutama dalam konteks risiko stroke dan kesehatan. Dataset yang digunakan dalam penelitian ini terdiri dari 12 atribut yang relevan, termasuk usia, jenis kelamin, riwayat hipertensi, dan Body Mass Index (BMI), yang semuanya merupakan faktor risiko signifikan yang dapat mempengaruhi kemungkinan terjadinya stroke. Dengan menerapkan teknik SMOTE, yang berfokus pada menciptakan contoh sintetik dari kelas minoritas untuk mencapai keseimbangan antara kelas positif dan negatif, penelitian ini berhasil mencapai akurasi prediksi yang cukup tinggi, yaitu 85,45% ketika menggunakan pembagian data sebesar 80:20. Selain itu, ketika data dibagi dengan rasio 70:30, akurasi yang dicapai adalah 85,24%. Hasil ini menunjukkan konsistensi dalam performa prediksi model SVM, yang mengindikasikan bahwa teknik ini efektif dalam meningkatkan kemampuan model untuk mengenali pola-pola dalam data, meskipun terdapat ketidakseimbangan yang signifikan. Melalui temuan ini, studi ini menyoroti betapa pentingnya penerapan teknik penyeimbangan data dalam meningkatkan akurasi prediksi pada model SVM. Hal ini menggarisbawahi bahwa perhatian terhadap pengolahan data yang tepat dalam analisis model pembelajaran mesin tidak hanya berkontribusi pada hasil yang lebih akurat, tetapi juga berfungsi sebagai langkah kritis dalam upaya pencegahan dan deteksi dini penyakit berbahaya seperti stroke【2】.


## 2. Business Understanding

Pengembangan model prediksi risiko stroke memiliki potensi yang sangat besar dalam meningkatkan efisiensi diagnosis, mendukung upaya pencegahan dini, dan memperkuat proses pengambilan keputusan medis. Dengan kemampuan untuk memberikan prediksi yang akurat, model ini dapat membantu berbagai pihak, seperti dokter, pasien, dan institusi kesehatan, dalam mengidentifikasi risiko stroke pada tahap awal. Hal ini memungkinkan implementasi langkah-langkah preventif yang diperlukan untuk mencegah serangan stroke yang berpotensi menyebabkan kecacatan yang parah atau bahkan kematian.
Bagi dokter, keberadaan model ini sangat berharga, karena dapat membantu mereka dalam menentukan prioritas penanganan pasien yang memiliki risiko tinggi. Dengan informasi yang lebih tepat, dokter dapat melakukan intervensi yang lebih cepat dan efektif. Bagi pasien, model prediksi ini memberikan peringatan dini tentang potensi risiko stroke, sehingga mendorong mereka untuk membuat perubahan gaya hidup yang lebih sehat, seperti menerapkan pola makan yang baik, berolahraga secara teratur, dan mengelola faktor risiko lainnya. Selain itu, bagi institusi kesehatan, penerapan model ini dapat mengakibatkan pengurangan biaya perawatan jangka panjang yang sering kali terkait dengan perawatan pasien stroke, yang tidak hanya menguntungkan secara finansial tetapi juga membantu dalam alokasi sumber daya yang lebih efisien. Dengan semua manfaat ini, pengembangan model prediksi risiko stroke bukan hanya sebuah inovasi teknologi, tetapi juga sebuah langkah penting dalam upaya meningkatkan kesehatan masyarakat secara keseluruhan.

### Problem Statements
Berdasarkan latar belakang tersebut, masalah yang dapat diselesaikan dalam proyek ini adalah:
- Bagaimana membangun model machine learning dan deep learning yang efektif untuk memprediksi risiko stroke berdasarkan data klinis dan demografis, dengan tingkat - akurasi yang tinggi?
- Algoritma apa yang memberikan performa terbaik untuk prediksi risiko stroke di antara model pembelajaran mesin dan mendalam yang digunakan?
- Bagaimana model ini dapat digunakan untuk mendukung penelitian kesehatan, seperti analisis tren faktor risiko stroke atau pengembangan sistem rekomendasi untuk pencegahan stroke berbasis data?

### Goals
Tujuan dari proyek ini meliputi:
- Membangun model prediksi risiko stroke berbasis machine learning dan deep learning dengan tingkat akurasi tinggi.
- Membandingkan performa berbagai algoritma untuk menentukan model terbaik dalam mendeteksi risiko stroke.
- Menghasilkan insight berbasis data yang mendukung penelitian kesehatan terkait pencegahan stroke, seperti mengidentifikasi tren demografis dan faktor risiko utama.


### Solution Statements
#### 1. Analisis Data dan Preprocessing
- Analisis Univariate dan Multivariate: Untuk memahami distribusi data, hubungan antar fitur, serta mendeteksi outlier. Contohnya adalah memeriksa kolerasi antara variabel seperti tekanan darah, kadar glukosa, dan BMI terhadap risiko stroke.
- Data Cleaning dan Normalisasi: Membersihkan data dari nilai yang hilang atau tidak valid, dan menormalkan fitur numerik untuk memastikan konsistensi dalam pelatihan model.

#### 2. Implementasi Model Machine Learning
- Membangun beberapa model menggunakan algoritma berikut:
    * Logistic Regression: Digunakan sebagai baseline untuk memahami pola hubungan antara variabel prediktor dan risiko stroke.
    * Random Forest: Algoritma ensemble yang dapat menangani data non-linear dan memberikan prediksi yang stabil dengan akurasi tinggi.
    * Support Vector Machine (SVM): Untuk memisahkan data dengan margin optimal, terutama pada dataset dengan dimensi tinggi.
    * Naive Bayes: Algoritma probabilistik yang efisien untuk tugas klasifikasi sederhana.
    * K-Nearest Neighbors (KNN): Membandingkan data baru dengan tetangga terdekatnya untuk klasifikasi risiko stroke.
    * XGBoost: Algoritma boosting yang memberikan performa tinggi dengan menangkap pola kompleks dalam data.

- Evaluasi Model:
    * Metrik seperti Akurasi, Precision, Recall, F1-Score, dan AUC-ROC akan digunakan untuk mengevaluasi model.

#### 3. Penerapan Algoritma Deep Learning
- Mengintegrasikan algoritma pembelajaran mendalam untuk meningkatkan prediksi:
    * Artificial Neural Network (ANN): Untuk memodelkan hubungan non-linear kompleks dalam data.
    * Recurrent Neural Network (RNN): Untuk menangkap pola temporal jika terdapat data berbasis waktu (misalnya riwayat kesehatan).
    * Long Short-Term Memory (LSTM): Untuk menangani pola temporal jangka panjang dengan mitigasi masalah vanishing gradient.

#### 4. Optimalisasi Model
- Penanganan Ketidakseimbangan Data: Dengan teknik seperti SMOTE atau ADASYN untuk meningkatkan performa model pada kelas minoritas (stroke).
- Hyperparameter Tuning: Menggunakan Grid Search atau Random Search untuk mencari parameter optimal seperti jumlah neuron, learning rate, dan jumlah estimators pada ensemble models.
- Cross-Validation: K-fold cross-validation untuk mengevaluasi model secara konsisten di seluruh dataset.

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | _Apple Quality_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| Maintainer | [fedesoriano ⚡](https://www.kaggle.com/fedesoriano) |
| License | Data files © Original Authors |
| Visibility | Publik |
| Tags | _Health, Education, Health Conditions, Public Health, Healthcare, Binary Classification_ |
| View | 1.38M |

![image](https://github.com/user-attachments/assets/2615011d-d8e4-4560-b0cb-4e56393f8614)

Gambar 1. Informasi Dataset

Dilihat dari _Gambar 1. Informasi Dataset_ dataset ini berisi informasi sebagai berikut ini : 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 5110 sample dengan 12 fitur.
- Dataset memiliki 3 fitur bertipe float64, 4 fitur bertipe o=int64 dan 5 fitur bertipe object.
- Terdapat 1 missing value dalam dataset pada kolom BMI.
### Variable - variable pada dataset
- id: Merupakan pengidentifikasi unik untuk setiap pasien, yang memungkinkan pengelolaan dan pelacakan data dengan lebih efisien.
- gender: Menunjukkan jenis kelamin pasien, yang dapat berupa "Laki-laki", "Perempuan", atau "Lainnya". Atribut ini penting untuk analisis perbedaan risiko berdasarkan gender.
- age: Menunjukkan usia pasien, yang merupakan faktor kunci dalam penilaian risiko stroke, karena kecenderungan serangan stroke meningkat seiring bertambahnya usia.
- hypertension: Indikator yang menunjukkan apakah pasien menderita hipertensi atau tidak, diwakili dengan nilai 0 jika tidak mengalami hipertensi dan 1 jika mengalami hipertensi. Hipertensi adalah salah satu faktor risiko utama untuk stroke.
- heart_disease: Menunjukkan apakah pasien memiliki penyakit jantung, dengan nilai 0 untuk tidak ada penyakit jantung dan 1 untuk memiliki penyakit jantung. Keberadaan penyakit jantung dapat meningkatkan risiko stroke secara signifikan.
- ever_married: Menunjukkan status pernikahan pasien, dengan opsi "Tidak" atau "Ya". Status pernikahan dapat berhubungan dengan berbagai faktor sosial dan emosional yang memengaruhi kesehatan.
- work_type: Kategori pekerjaan pasien, yang bisa berupa "anak-anak", "Pekerjaan Pemerintah", "Belum pernah bekerja", "Swasta", atau "Wiraswasta". Tipe pekerjaan dapat memberikan wawasan tentang paparan stres dan faktor risiko terkait pekerjaan.
- Residence_type: Menunjukkan jenis tempat tinggal pasien, apakah berada di "Pedesaan" atau "Perkotaan". Lingkungan tempat tinggal dapat mempengaruhi akses terhadap pelayanan kesehatan yang berkualitas.
- avg_glucose_level: Menyediakan informasi tentang rata-rata kadar glukosa dalam darah pasien. Kadar glukosa yang tinggi dapat menjadi indikator risiko penyakit metabolik yang berkaitan dengan stroke.
- bmi: Body Mass Index (Indeks Massa Tubuh) pasien, yang merupakan ukuran untuk menentukan apakah seseorang memiliki berat badan yang sehat, berlebih, atau kurang. BMI yang tidak normal dapat berkontribusi pada risiko penyakit cardiovascular, termasuk stroke.
- smoking_status: Menunjukkan status merokok pasien, yang dapat berupa "pernah merokok", "tidak pernah merokok", "sedang merokok" atau "Tidak Diketahui". Merokok adalah faktor risiko yang signifikan untuk banyak penyakit, termasuk stroke.
- stroke: Indikator yang menunjukkan apakah pasien pernah mengalami stroke, diwakili dengan nilai 1 jika pernah mengalami stroke dan 0 jika tidak. Atribut ini merupakan hasil yang ingin diprediksi dalam model analisis risiko stroke.
- Informasi di atas mencakup berbagai faktor demografis, kesehatan, dan gaya hidup yang relevan dalam menentukan risiko stroke, dan dapat digunakan untuk memfasilitasi pengembangan model prediksi yang lebih akurat.

### EDA - Univariate Analysis

![image](https://github.com/user-attachments/assets/5faa91e0-f930-488d-a0fc-002ab3fa8f1c)

Gambar 2. Informasi Dataset

Gambar 2 merupakan informasi mengenai dataset yang sigunakan :
- Kolom id berfungsi sebagai pengidentifikasi unik bagi setiap individu, namun statistik deskriptif seperti rata-rata (mean), nilai minimum (min), dan maksimum (max) tidak begitu relevan untuk analisis kolom ini. Kolom age menunjukkan usia individu, dengan rata-rata usia sebesar 43,23 tahun. Usia termuda dalam dataset tercatat adalah 0,08 tahun (mungkin bayi), sementara usia tertua adalah 82 tahun. Distribusi usia juga dapat dilihat dari kuartil, di mana 50% individu berusia 45 tahun atau lebih muda.
- Untuk kolom hypertension, yang merupakan variabel biner (0 atau 1) yang menunjukkan apakah individu memiliki hipertensi, didapatkan rata-rata sebesar 0,097, yang berarti sekitar 9,7% individu dalam dataset memiliki hipertensi. Demikian juga pada kolom heart_disease, yang menunjukkan riwayat penyakit jantung, dengan rata-rata 0,054, mengindikasikan bahwa hanya sekitar 5,4% individu yang mengalami kondisi ini.
- Kolom avg_glucose_level mencerminkan rata-rata kadar glukosa darah individu dengan rata-rata sebesar 106,15 mg/dL, yang termasuk dalam kisaran normal. Namun, nilai maksimum mencapai 271,74 mg/dL, yang bisa mengindikasikan adanya individu dengan diabetes atau hiperglikemia. Median kadar glukosa (50%) adalah 91,89 mg/dL, menunjukkan bahwa setengah dari data berada di bawah nilai tersebut.
- Sedangkan kolom bmi atau Body Mass Index menunjukkan rata-rata BMI sebesar 28,89, yang mendekati kategori overweight berdasarkan pedoman WHO. Nilai maksimum BMI tercatat 97,6, yang menunjukkan kemungkinan adanya outlier, baik karena kesalahan data atau kasus ekstrem obesitas. Penting untuk dicatat bahwa terdapat data kosong, dengan 4.909 data valid dari total 5.110, sehingga perlu penanganan khusus untuk analisis lanjutan.
- Pada kolom stroke, yang merupakan variabel target biner menunjukkan apakah individu pernah mengalami stroke, didapatkan rata-rata 0,048, menandakan sekitar 4,87% individu dalam dataset pernah mengalami stroke. Dari kesimpulan awal, sebagian besar individu dalam dataset tidak memiliki hipertensi, penyakit jantung, ataupun riwayat stroke. Data seperti BMI yang memiliki missing values perlu ditangani sebelum model pemodelan. Selain itu, nilai maksimum pada beberapa kolom, termasuk avg_glucose_level dan bmi, menunjukkan adanya kemungkinan outlier yang dapat memengaruhi hasil analisis atau model prediksi. Faktor usia, hipertensi, penyakit jantung, dan kadar glukosa darah terlihat relevan untuk analisis risiko stroke.

![image](https://github.com/user-attachments/assets/73e7a5e0-7a33-4360-b1eb-32b55af536ce)
Gambar 3. Persebaran data pada dataset stroke

Gambar 3 merupakan visualisasi exploratory data analysis dari persebaran data pada dataset yang digunakan pada project ini adalah stroke dataset. Dapat dilihat pada gambar diatas terlihat visualisasi dari sebaran data setiap kolom yang terdapat pada dataset. Adapun penjelasan dari sebaran data dari gambar diatas adalah sebagai berikut :

- Distribusi data BMI cenderung mirip distribusi normal dengan puncak sekitar 20-30. Namun, terdapat beberapa outlier dengan BMI di atas 60 yang perlu dipertimbangkan untuk preprocessing agar model prediksi menjadi lebih efektif.
- Sebagian besar data tingkat glukosa terkonsentrasi antara 50 hingga 150, dengan adanya outlier yang mencapai lebih dari 200. Distribusi ini cenderung right-skewed, yang menunjukkan bahwa ada sebagian kecil individu yang memiliki tingkat glukosa sangat tinggi.
- Distribusi usia terlihat cukup merata, dengan peningkatan pada kelompok usia 40-60 tahun. Hal ini menunjukkan variasi yang baik dalam rentang usia, meskipun terdapat potensi bias yang lebih condong kepada kelompok dewasa dan lansia.
- Dalam dataset, sekitar 95% individu tidak mengalami stroke (label 0), sedangkan hanya 5% yang mengalami stroke (label 1). Ketidakseimbangan kelas yang signifikan ini penting untuk diperhatikan dalam proses pembuatan model prediksi agar hasilnya lebih akurat. Pada project ini akan diterapkan SMOTE untuk memperbaiki keseimbangan kelas.
- Hampir 90% individu tidak memiliki hipertensi (label 0), sementara sekitar 10% memiliki hipertensi (label 1). Distribusi yang ada menunjukkan ketidakseimbangan data yang perlu dicermati saat membangun model analisis.
- Hanya sekitar 5% individu dalam dataset yang memiliki riwayat penyakit jantung (label 1), sementara sisanya tidak mengalami penyakit jantung (label 0). Ketidakseimbangan kelas ini juga perlu diperhatikan untuk meningkatkan kinerja model prediksi.
- Mayoritas individu dalam dataset pernah menikah (label "Yes") dibandingkan yang belum pernah menikah (label "No"). Fitur ini mungkin memiliki hubungan dengan usia, mengingat orang yang lebih tua cenderung lebih mungkin sudah menikah.
- Distribusi gender dalam dataset hampir seimbang antara laki-laki (Male) dan perempuan (Female), dengan jumlah perempuan sedikit lebih tinggi. Terdapat juga kelas tambahan "Other," tetapi jumlahnya sangat kecil dan tidak signifikan.
- Data menunjukkan distribusi yang hampir merata antara individu yang tinggal di daerah urban dan rural, menandakan keseimbangan dalam fitur ini.
Terdapat ketidakseimbangan yang signifikan pada fitur target (stroke) serta fitur terkait seperti hypertension dan heart disease, yang harus diperhatikan saat melatih model. Penggunaan metode SMOTE atau class weighting bisa dipertimbangkan untuk mengatasi masalah ini. Analisis korelasi antar fitur juga diperlukan untuk memahami hubungan, khususnya antara age, hypertension, heart disease, dan stroke.

### EDA - Multivariate Analysis

![image](https://github.com/user-attachments/assets/e01f9457-e410-4741-b2d8-07d17e5f0e2d)
Gambar 4. Analisis Multivariate

![image](https://github.com/user-attachments/assets/27ba219a-f217-4b60-8cbf-a61a4dd2e70e)
Gambar 5. Analisis Matriks Korelasi

Pada Gambar 4 Analisis Multivariat, dengan menggunakan fungsi pairplot dari library seaborn, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada pola sebaran data grafik pairplot. Dan Pada Gambar 5 Analisis Matriks Korelasi, merupakan Correlation Matrix menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur umur memiliki skor korelasi yang cukup besar 0.28 dengan fitur hipertensi, dan juga fitur berat badan yang cukup besar 0.33 dengan fitur umur.

## Data Preparation
Berikut merupakan data preparation yang diterapkan pada project ini :

1. Data Gathering
Pada tahap ini, data diimpor dengan hati-hati agar dapat dibaca dan diproses dengan baik menggunakan dataframe dari library Pandas. Proses ini penting untuk memastikan data yang dikumpulkan dapat diolah secara efisien dan akurat.

2. Data Assessing
Setelah data berhasil dikumpulkan, langkah ini melibatkan beberapa pengecekan untuk memastikan kualitas data, meliputi:
- Duplicate Data
Pengecekan dilakukan untuk mencari data yang serupa atau identik dengan data lainnya dalam dataset. Mengidentifikasi dan menghapus duplikasi penting untuk menghindari bias dalam analisis.
- Missing Value
Pengecekan dilakukan untuk menemukan data atau informasi yang hilang atau tidak tersedia. Menangani missing value sangat penting agar integritas dataset tetap terjaga. Pada project ini terdapat missing value pada kolom BMI, cara untuk mengisi missing value yang dilakukan pada pre-processingnya ada dengan mengisi data dengan rata-rata dari BMI yang terdapat pada dataset.
- Outlier
Analisis terhadap data yang menyimpang dari pola atau rata-rata sekumpulan data dilakukan untuk menentukan apakah ada outlier. Keberadaan outlier dapat memberikan informasi penting atau mengganggu model prediksi yang dikembangkan.

3. Data Cleaning
Pada tahap ini, beberapa langkah diambil untuk membersihkan dan menyiapkan data agar siap untuk analisis lebih lanjut:
- Converting Column Type
Tipe data pada kolom tertentu diubah agar sesuai dengan kebutuhan analisis. Misalnya, kolom yang seharusnya berisi data numerik tetapi terformat sebagai string perlu diubah untuk memungkinkan operasi matematika.
- Train Test Split
Dataset dibagi menjadi dua bagian: satu untuk data latih dan satu lagi untuk data uji. Pembagian ini sangat penting untuk menguji model secara independen dengan data yang tidak pernah dilihat sebelumnya, sehingga kemampuan generalisasi model dapat divalidasi.
- Normalization
Data ditransformasi ke dalam skala yang seragam, sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding. Hal ini bertujuan untuk mengurangi bias akibat perbedaan skala antar atribut dan meningkatkan kinerja algoritma pembelajaran mesin.
- SMOTE (Synthetic Minority Over-sampling Technique)
Ketidakseimbangan kelas dalam dataset, seperti dalam kasus fitur target (misalnya stroke), SMOTE diterapkan. Teknik ini menghasilkan contoh sintetik dari kelas minoritas dengan cara menginterpolasi antara contoh yang ada, sehingga membantu model untuk belajar dari data yang lebih seimbang.

Proses preprocessing dalam project ini akan dijelaskan sebagai berikut:

1. Pemeriksaan Missing Values
Fungsi data.isnull().sum() digunakan untuk menghitung jumlah data yang hilang (missing values) di setiap kolom dataset. Langkah ini penting untuk mengetahui apakah ada nilai kosong yang perlu ditangani sebelum memproses data lebih lanjut.

2. Pemilahan Fitur (X) dan Label (Y)
Dataset dipisahkan menjadi fitur independen x (dari kolom kedua hingga sebelum kolom terakhir) dan target label y (kolom terakhir). Hal ini dilakukan dengan menggunakan iloc dari pandas. Proses ini adalah langkah awal untuk membangun model machine learning di mana fitur digunakan untuk memprediksi label.

3. Encoding Kategorikal (One-Hot Encoding)
Untuk mengonversi fitur kategorikal menjadi bentuk numerik yang dapat diproses oleh algoritma machine learning, digunakan One-Hot Encoding.
Kolom 0, 5, dan 9 yang berisi data kategorikal diterjemahkan menjadi representasi biner menggunakan ColumnTransformer dengan OneHotEncoder. Kolom lainnya tetap diproses tanpa perubahan (remainder='passthrough).

4. Encoding Label (Label Encoding):
Beberapa kolom spesifik diubah menjadi representasi numerik diskrit menggunakan Label Encoding. Ini dilakukan dengan LabelEncoder dari sklearn. Langkah ini berguna untuk kategori yang hanya memiliki dua atau sedikit label unik, seperti "Yes/No".

5. Pengecekan Dimensi Data
Ukuran data hasil preprocessing diperiksa dengan x.shape dan y.shape untuk memastikan data telah diproses dengan benar. Hal ini memastikan bahwa dimensi data sesuai dengan yang diharapkan sebelum melanjutkan ke pelatihan model.

Proses di atas bertujuan untuk menyiapkan data dalam format yang dapat diproses oleh algoritma machine learning. Fitur kategorikal diubah menjadi representasi numerik, fitur target dipisahkan, dan potensi permasalahan data seperti missing values diperiksa untuk meminimalkan error selama pelatihan model. Pada proyek ini digunakan Train Test Split pada library sklearn.model_selection untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 20:80 dan random state sebesar 0. Semua proses ini diperlukan dalam rangka membuat model yang baik.

## Modeling
Pada project ini menggunakan 8 algoritma machine learning dan 3 algoritma deep learning, yang diantaranya sebagai berikut :



## Evaluation







## Daftar Pustaka

1. None Indah Werdiningsih, Endah Purwanti, Iin Mardiyana, Arum Tiyas Handayani, Kharristantie Sekarlangit Suryadewi, Endang Nurjanah, Fildzah Akhlaqulkarimah, Naurah Hedy Pramiyas, & Almas, F. (2023). Analisis Prediksi Stroke Menggunakan Pendekatan Decision Tree dengan Seleksi Fitur dan Neural Network. Jurnal Sistem Cerdas, 6(3), 213–221. https://doi.org/10.37396/jsc.v6i3.310

‌2. Wulandari, E., & Arita Witanti. (2024). The CLASSIFICATION OF STROKE PREDICTION USING THE SUPPORT VECTOR MACHINE (SVM) METHOD. JATISI (Jurnal Teknik Informatika Dan Sistem Informasi), 11(3). https://doi.org/10.35957/jatisi.v11i3.8044

‌3. fedesoriano. (2021). Stroke Prediction Dataset. Kaggle.com. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

‌4. 
