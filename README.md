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
Dengan pendekatan ini, angka kejadian stroke dapat ditekan secara signifikan, mengurangi beban ekonomi pada keluarga dan sistem kesehatan【1】【2】.


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
    * Metrik seperti Akurasi, Precision, Recall dan F1-Score digunakan untuk mengevaluasi model, Metrik evaluasi yang ditekankan pada project ini adalah metrik evaluasi dengan menggunakan akurasi

#### 3. Penerapan Algoritma Deep Learning
- Mengintegrasikan algoritma pembelajaran mendalam untuk meningkatkan prediksi:
    * Artificial Neural Network (ANN): Untuk memodelkan hubungan non-linear kompleks dalam data.
    * Recurrent Neural Network (RNN): Untuk menangkap pola temporal jika terdapat data berbasis waktu (misalnya riwayat kesehatan).
    * Long Short-Term Memory (LSTM): Untuk menangani pola temporal jangka panjang dengan mitigasi masalah vanishing gradient.

#### 4. Optimalisasi Model
- Penanganan Ketidakseimbangan Data: Dengan teknik seperti SMOTE untuk meningkatkan performa model pada kelas minoritas (stroke).
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

Tabel 1. Informasi Dataset

Dilihat dari _Tabel 1. Informasi Dataset_ dataset ini berisi informasi sebagai berikut ini : 
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
Informasi di atas mencakup berbagai faktor demografis, kesehatan, dan gaya hidup yang relevan dalam menentukan risiko stroke, dan dapat digunakan untuk memfasilitasi pengembangan model prediksi yang lebih akurat.

### EDA - Univariate Analysis

![image](https://github.com/user-attachments/assets/5faa91e0-f930-488d-a0fc-002ab3fa8f1c)

Gambar 1. Informasi Dataset

Gambar 1 merupakan informasi mengenai dataset yang sigunakan :
- Kolom id berfungsi sebagai pengidentifikasi unik bagi setiap individu, namun statistik deskriptif seperti rata-rata (mean), nilai minimum (min), dan maksimum (max) tidak begitu relevan untuk analisis kolom ini. Kolom age menunjukkan usia individu, dengan rata-rata usia sebesar 43,23 tahun. Usia termuda dalam dataset tercatat adalah 0,08 tahun (mungkin bayi), sementara usia tertua adalah 82 tahun. Distribusi usia juga dapat dilihat dari kuartil, di mana 50% individu berusia 45 tahun atau lebih muda.
- Untuk kolom hypertension, yang merupakan variabel biner (0 atau 1) yang menunjukkan apakah individu memiliki hipertensi, didapatkan rata-rata sebesar 0,097, yang berarti sekitar 9,7% individu dalam dataset memiliki hipertensi. Demikian juga pada kolom heart_disease, yang menunjukkan riwayat penyakit jantung, dengan rata-rata 0,054, mengindikasikan bahwa hanya sekitar 5,4% individu yang mengalami kondisi ini.
- Kolom avg_glucose_level mencerminkan rata-rata kadar glukosa darah individu dengan rata-rata sebesar 106,15 mg/dL, yang termasuk dalam kisaran normal. Namun, nilai maksimum mencapai 271,74 mg/dL, yang bisa mengindikasikan adanya individu dengan diabetes atau hiperglikemia. Median kadar glukosa (50%) adalah 91,89 mg/dL, menunjukkan bahwa setengah dari data berada di bawah nilai tersebut.
- Sedangkan kolom bmi atau Body Mass Index menunjukkan rata-rata BMI sebesar 28,89, yang mendekati kategori overweight berdasarkan pedoman WHO. Nilai maksimum BMI tercatat 97,6, yang menunjukkan kemungkinan adanya outlier, baik karena kesalahan data atau kasus ekstrem obesitas. Penting untuk dicatat bahwa terdapat data kosong, dengan 4.909 data valid dari total 5.110, sehingga perlu penanganan khusus untuk analisis lanjutan.
- Pada kolom stroke, yang merupakan variabel target biner menunjukkan apakah individu pernah mengalami stroke, didapatkan rata-rata 0,048, menandakan sekitar 4,87% individu dalam dataset pernah mengalami stroke. Dari kesimpulan awal, sebagian besar individu dalam dataset tidak memiliki hipertensi, penyakit jantung, ataupun riwayat stroke. Data seperti BMI yang memiliki missing values perlu ditangani sebelum model pemodelan. Selain itu, nilai maksimum pada beberapa kolom, termasuk avg_glucose_level dan bmi, menunjukkan adanya kemungkinan outlier yang dapat memengaruhi hasil analisis atau model prediksi. Faktor usia, hipertensi, penyakit jantung, dan kadar glukosa darah terlihat relevan untuk analisis risiko stroke.

![image](https://github.com/user-attachments/assets/73e7a5e0-7a33-4360-b1eb-32b55af536ce)
Gambar 2. Persebaran data pada dataset stroke

Gambar 2 merupakan visualisasi exploratory data analysis dari persebaran data pada dataset yang digunakan pada project ini adalah stroke dataset. Dapat dilihat pada gambar diatas terlihat visualisasi dari sebaran data setiap kolom yang terdapat pada dataset. Adapun penjelasan dari sebaran data dari gambar diatas adalah sebagai berikut :

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
Gambar 3. Analisis Multivariate

![image](https://github.com/user-attachments/assets/27ba219a-f217-4b60-8cbf-a61a4dd2e70e)
Gambar 4. Analisis Matriks Korelasi

Pada Gambar 3 Analisis Multivariat, dengan menggunakan fungsi pairplot dari library seaborn, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada pola sebaran data grafik pairplot. Dan Pada Gambar 4 Analisis Matriks Korelasi, merupakan Correlation Matrix menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur umur memiliki skor korelasi yang cukup besar 0.28 dengan fitur hipertensi, dan juga fitur berat badan yang cukup besar 0.33 dengan fitur umur.

## Data Preparation
Berikut merupakan data preparation yang diterapkan pada project ini :

1. Data Gathering
Pada tahap ini, data diimpor dengan hati-hati agar dapat dibaca dan diproses dengan baik menggunakan dataframe dari library Pandas. Proses ini penting untuk memastikan data yang dikumpulkan dapat diolah secara efisien dan akurat.

2. Pemeriksaan Missing Values
Langkah awal adalah memastikan tidak ada data yang hilang (missing values) pada dataset. Pemeriksaan dilakukan untuk mengetahui apakah terdapat kolom atau baris dengan nilai kosong. Jika terdapat missing values, langkah penanganan seperti imputasi (mengisi nilai kosong dengan rata-rata) perlu diterapkan untuk menjaga integritas dataset. Dalam kasus ini, hasil pemeriksaan menunjukkan tidak ada data yang hilang, sehingga tidak diperlukan langkah penanganan lebih lanjut untuk missing values. Pada proyek ini diterapkan pada kolom BMI yang memiliki missing value dan diisi oleh nilai rata-rata BMI.

3. Pemilahan Fitur (X) dan Label (Y)
Dataset dipisahkan menjadi dua bagian utama:
   - Fitur independen (X): Kolom yang digunakan untuk memprediksi hasil (dari kolom kedua hingga sebelum kolom terakhir).
   - Target label (Y): Kolom terakhir, yang merupakan output yang ingin diprediksi oleh model. Langkah ini penting untuk memisahkan variabel yang digunakan dalam analisis dan variabel target yang akan diprediksi.

4. Encoding Kategorikal
Fitur kategorikal, yang berupa data non-numerik seperti nama atau kategori, diubah menjadi representasi numerik menggunakan One-Hot Encoding. Pada dataset ini, kolom tertentu yang memiliki nilai kategorikal diterjemahkan menjadi representasi biner. Hal ini dilakukan untuk memastikan algoritma machine learning dapat memproses data tersebut. One-Hot Encoding diterapkan pada kolom yang mengandung kategori, sementara kolom lainnya tetap dipertahankan.

5. Encoding Label
Untuk kolom yang memiliki kategori dengan jumlah label unik sedikit (seperti "Yes/No"), digunakan Label Encoding. Teknik ini mengonversi nilai kategorikal menjadi angka diskrit (misalnya, "Yes" menjadi 1 dan "No" menjadi 0). Langkah ini mempermudah algoritma machine learning dalam memproses data dengan kategori sederhana.

6. Pengecekan Dimensi Data
Setelah proses encoding, dimensi data diperiksa untuk memastikan bahwa jumlah fitur dan label sudah sesuai dengan yang diharapkan. Hal ini memastikan data telah diproses dengan benar dan siap untuk digunakan dalam tahap berikutnya.

7. Split Dataset
Dataset dibagi menjadi dua subset:
   - Data latih (training set): Digunakan untuk melatih model agar dapat mengenali pola dalam data.
   - Data uji (test set): Digunakan untuk mengevaluasi performa model pada data baru yang belum pernah dilihat sebelumnya.
Pembagian dilakukan dengan rasio 80:20, di mana 80% digunakan untuk pelatihan dan 20% untuk pengujian. Random state juga digunakan untuk memastikan hasil pembagian dataset konsisten di setiap eksekusi.

8. Feature Scaling
Data diformulasikan ulang agar semua fitur berada dalam skala yang seragam menggunakan StandardScaler. Teknik ini menstandarkan nilai-nilai fitur dengan mengurangi rata-rata dan membaginya dengan standar deviasi. Proses ini penting untuk mengurangi bias akibat perbedaan skala antar fitur dan meningkatkan efisiensi algoritma.

9. Penanganan Ketidakseimbangan Data dengan SMOTE
Ketidakseimbangan kelas sering menjadi masalah dalam dataset, terutama jika salah satu kelas target jauh lebih sedikit dibandingkan kelas lainnya. Dalam kasus ini, digunakan SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan data. Teknik ini menghasilkan sampel sintetik dari kelas minoritas dengan cara menginterpolasi antara contoh yang ada. Hasilnya adalah dataset yang lebih seimbang, memungkinkan model untuk mempelajari pola dari kedua kelas secara lebih efektif.
   - Sebelum penerapan SMOTE, ditampilkan jumlah data pada masing-masing kelas.
   - Setelah SMOTE, dataset di-resample sehingga jumlah data untuk setiap kelas menjadi seimbang.

## Modeling
Pada project ini menggunakan 8 algoritma machine learning dan 3 algoritma deep learning dan juga terdapat dua kondisi dengan tidak menetapkan hyperparameter dan menerapkan hyperparameter tuning menggunakan gridsearch, yang diantaranya sebagai berikut :

### 1. Logistic Regression

**Logistic Regression** adalah metode statistik yang digunakan untuk klasifikasi biner dengan memodelkan probabilitas suatu peristiwa terjadi. Metode ini menghasilkan fungsi sigmoid yang mengubah output regresi linier menjadi nilai antara 0 dan 1.

#### Kelebihan
- **Mudah Diinterpretasikan**  
  Memberikan probabilitas yang jelas untuk klasifikasi, memudahkan pemahaman hasil.
- **Cepat dan Efisien**  
  Proses pelatihan cepat, efektif untuk dataset kecil hingga menengah.
- **Penerapan Luas**  
  Sering digunakan dalam bidang medis, pemasaran, dan ilmu sosial.

#### Kekurangan
- **Asumsi Linearitas**  
  Kurang efektif untuk data dengan hubungan non-linier tanpa transformasi fitur.
- **Sensitif Terhadap Outlier**  
  Kinerja dapat terpengaruh oleh pencilan dalam dataset.

#### Parameter
- **`C`**  
  Regularisasi invers (1/λ). Semakin kecil nilai C, semakin kuat regularisasi untuk menghindari overfitting.
- **`penalty`**  
  Jenis penalti pada koefisien model:  
  - `l1`: Penalti Lasso.  
  - `l2`: Penalti Ridge.  
  - `elasticnet`: Kombinasi penalti L1 dan L2.  
  - `none`: Tanpa penalti.  
- **`solver`**  
  Algoritma optimasi (contoh: `liblinear` untuk dataset kecil).
- **`random_state`**  
  Untuk memastikan hasil dapat direplikasi.

---

### 2. K-Nearest Neighbors (KNN)

**KNN** adalah algoritma pembelajaran yang digunakan untuk klasifikasi dan regresi. Metode ini mencari `k` tetangga terdekat di ruang fitur dan menggunakan mayoritas kelas tetangga untuk menentukan kelas data baru.

#### Kelebihan
- **Intuitif dan Sederhana**  
  Konsep dasar mudah dipahami dan diimplementasikan.
- **Non-parametrik**  
  Tidak memerlukan proses pelatihan rumit.

#### Kekurangan
- **Lambat untuk Dataset Besar**  
  Kinerja menurun saat menghitung jarak terhadap seluruh dataset.
- **Sensitif terhadap Fitur**  
  Memerlukan skala fitur agar hasil tidak terdistorsi.

#### Parameter
- **`n_neighbors`**  
  Jumlah tetangga yang dijadikan referensi.
- **`metric`**  
  Metode pengukuran jarak, seperti Euclidean atau Manhattan.
- **`weights`**  
  Menentukan apakah semua tetangga memiliki bobot sama atau berdasarkan jarak.
- **`p`**  
  Parameter untuk Minkowski. Nilai `p=1` untuk jarak Manhattan, `p=2` untuk jarak Euclidean.

---

### 3. Support Vector Machine (SVM)

**SVM** adalah algoritma klasifikasi yang mencari hyperplane optimal untuk memisahkan data dari dua kelas. Dengan kernel, SVM juga dapat memisahkan data non-linier.

#### Kelebihan
- **Efektif pada Dimensi Tinggi**  
  Sangat baik dalam ruang fitur tinggi.
- **Tahan Terhadap Overfitting**  
  Cenderung menghasilkan model yang generalisasi dengan baik.

#### Kekurangan
- **Lambat pada Dataset Besar**  
  Waktu pelatihan besar karena kompleksitas perhitungan.
- **Perlu Tuning Parameter yang Hati-hati**  
  Memerlukan pemilihan parameter `C` dan `gamma` yang cermat.

#### Parameter
- **`C`**  
  Mengontrol kesalahan yang diizinkan dalam klasifikasi.
- **`kernel`**  
  Jenis kernel yang digunakan, seperti `linear`, `rbf`, atau `poly`.
- **`gamma`**  
  Besarnya pengaruh data individu pada keputusan klasifikasi.
- **`degree`**  
  Tingkat polinomial untuk kernel polynomial.
- **`random_state`**  
  Untuk hasil konsisten.

---

### 4. Naive Bayes (Bernoulli)

**Naive Bayes Bernoulli** adalah algoritma klasifikasi probabilistik berdasarkan teorema Bayes, khusus untuk dataset biner.

#### Kelebihan
- **Cepat dan Efisien**  
  Proses pelatihan dan prediksi sangat cepat.
- **Cocok untuk Data Diskrit**  
  Ideal untuk teks atau data biner.

#### Kekurangan
- **Asumsi Independensi Fitur**  
  Jarang terpenuhi dalam kenyataan, dapat menurunkan akurasi.
- **Tidak Cocok untuk Data Kontinu**  
  Kualitas prediksi berkurang jika diterapkan pada dataset kontinu.

#### Parameter
- **`alpha`**  
  Parameter smoothing Laplace untuk menghindari probabilitas nol.
- **`binarize`**  
  Ambang batas untuk binarisasi data input.

---

### 5. Naive Bayes (Gaussian)

**Naive Bayes Gaussian** adalah variasi dari Naive Bayes untuk data kontinu dengan asumsi fitur mengikuti distribusi Gaussian.

#### Kelebihan
- **Efisien untuk Data Kontinu**  
  Cocok untuk data numerik.
- **Tidak Memerlukan Tuning Parameter yang Rumit**  
  Hanya memerlukan estimasi rata-rata dan varians.

#### Kekurangan
- **Sensitif terhadap Deviations dari Asumsi Gaussian**  
  Kinerja menurun jika data tidak terdistribusi normal.
- **Asumsi Independensi Fitur**  
  Sama seperti Bernoulli.

#### Parameter
- **`var_smoothing`**  
  Menambahkan nilai kecil pada varians untuk menghindari pembagian nol.

---

### 6. Decision Tree

**Decision Tree** menggunakan struktur pohon untuk membuat keputusan. Setiap node mewakili fitur, cabang adalah keputusan, dan daun adalah hasil klasifikasi.

#### Kelebihan
- **Mudah Diinterpretasikan**  
  Visualisasi pohon membantu pemahaman proses keputusan.
- **Tidak Memerlukan Normalisasi Fitur**  
  Dapat menangani fitur numerik dan kategorikal.

#### Kekurangan
- **Rentan terhadap Overfitting**  
  Membutuhkan pemangkasan atau regulasi.
- **Sensitif terhadap Perubahan Data**  
  Perubahan kecil pada data dapat mengubah struktur pohon secara signifikan.

#### Parameter
- **`max_depth`**  
  Kedalaman maksimum pohon.
- **`min_samples_split`**  
  Minimum sampel untuk membagi node.
- **`criterion`**  
  Ukuran kualitas pemisahan (Gini impurity atau entropy).
- **`splitter`**  
  Metode pemilihan split (`best` atau `random`).

---

### 7. Random Forest

**Random Forest** adalah metode ensemble yang menggunakan banyak decision tree untuk meningkatkan akurasi dan mengurangi overfitting.

#### Kelebihan
- **Robust terhadap Overfitting**  
  Lebih stabil dibandingkan decision tree tunggal.
- **Efektif untuk Data Besar**  
  Bekerja baik pada dataset besar.

#### Kekurangan
- **Kurang Interpretable**  
  Kompleksitas tinggi dibandingkan decision tree tunggal.
- **Lambat untuk Dataset Besar**  
  Waktu pelatihan relatif lama.

#### Parameter
- **`n_estimators`**  
  Jumlah pohon dalam ensemble.
- **`criterion`**  
  Fungsi split (Gini atau entropy).
- **`max_depth`**  
  Kedalaman maksimum pohon.
- **`min_samples_split`**  
  Sama seperti Decision Tree.

---

### 8. XGBoost

**XGBoost** adalah algoritma boosting yang memperbaiki kelemahan model sebelumnya dengan optimasi gradient boosting.

#### Kelebihan
- **Cepat dan Efisien**  
  Mendukung optimisasi paralel.
- **Penanganan Nilai Hilang**  
  Secara otomatis menangani nilai hilang.

#### Kekurangan
- **Memerlukan Tuning Parameter yang Kompleks**  
  Butuh pemilihan parameter yang hati-hati.
- **Risiko Overfitting**  
  Rentan terhadap overfitting pada dataset besar.

#### Parameter
- **`learning_rate`**  
  Ukuran langkah pembelajaran.
- **`max_depth`**  
  Kedalaman maksimum pohon.
- **`n_estimators`**  
  Jumlah iterasi boosting.

---

### 9. Artificial Neural Network (ANN)

**ANN** adalah model pembelajaran mesin yang terinspirasi oleh jaringan saraf biologis, dengan lapisan input, tersembunyi, dan output.

#### Kelebihan
- **Fleksibel untuk Berbagai Tipe Data**  
  Cocok untuk klasifikasi, regresi, dan lainnya.
- **Efektif untuk Hubungan Non-Linier**  
  Mampu memodelkan hubungan kompleks.

#### Kekurangan
- **Membutuhkan Banyak Data**  
  Memerlukan dataset besar untuk pelatihan.
- **Waktu Pelatihan Panjang**  
  Memakan waktu cukup lama untuk pelatihan.

#### Parameter
- **`layers`**  
  Struktur jaringan, termasuk jumlah layer.
- **`activation`**  
  Fungsi aktivasi seperti ReLU atau sigmoid.
- **`optimizer`**  
  Metode optimasi, seperti Adam atau SGD.

---

### 10. Recurrent Neural Network (RNN)

**RNN** adalah jaringan saraf untuk data sekuensial, menggunakan lapisan yang memiliki memori internal.

#### Kelebihan
- **Efektif untuk Data Sekuensial**  
  Cocok untuk teks dan time series.
- **Memori Jangka Panjang**  
  Dengan LSTM/GRU, mampu mengingat dependensi jangka panjang.

#### Kekurangan
- **Masalah Vanishing Gradient**  
  Kinerja menurun jika sekuens terlalu panjang.
- **Waktu Pelatihan Lama**  
  Butuh waktu lebih lama dibandingkan model tradisional.

--- 

### 11. Long Short-Term Memory (LSTM)

**LSTM** adalah varian dari RNN yang dirancang untuk mengatasi masalah *vanishing gradient* dengan memanfaatkan memori jangka panjang. LSTM memiliki sel memori yang memungkinkan informasi disimpan lebih lama.

#### Kelebihan
- **Mengatasi Masalah Vanishing Gradient**  
  Struktur unik LSTM memungkinkan pemodelan ketergantungan jangka panjang dengan efektif.
- **Cocok untuk Data Kompleks**  
  Ideal untuk analisis data sekuensial, seperti prediksi teks, bahasa alami, dan video.

#### Kekurangan
- **Waktu Pelatihan yang Panjang**  
  Membutuhkan waktu lebih banyak untuk pelatihan dibandingkan dengan model biasa.
- **Butuh Sumber Daya Komputasi Besar**  
  Memerlukan lebih banyak memori dan unit perhitungan dibandingkan model sederhana.

#### Parameter
- **`units`**  
  Jumlah unit dalam layer LSTM.
- **`dropout`**  
  Pengaturan dropout untuk mengurangi *overfitting* di antara waktu.
- **`return_sequences`**  
  Menentukan apakah agar setiap langkah waktu menghasilkan keluaran, cocok untuk data sekuensial lebih lanjut.


## Evaluation
Dalam tahap evaluasi pada proses pembuatan project ini, metrik yang digunakan adalah `accuracy`. Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Rumus:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

*Penjelasan*
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy 11 buah model yang latih:

| Model | Accuracy |
| ------ | ------ |
| Logistic Regression | 0.77 |
| KNN  | 0.83 |
| SVM | 0.79 |
| Naive Bayes (Bernoulli) | 0.60 |
| Naive Bayes (Gaussian) | 0.19 |
| Decision Tree | 0.86 |
| Random Forest | 0.90 |
| XGBOOST | 0.91 |
| ANN | 0.84 |
| RNN | 0.89 |
| LSTM | 0.90 |

Tabel 2. Hasil Accuracy Machine Learning dan Deep Learning tanpa hyperparameter tuning

Tabel 2 merupakan hasil dari akurasi pada model machine learning dan deep learning tanpa hyperparameter tuning. Pada evaluasi model, Logistic Regression mencapai akurasi 77%, menunjukkan performa yang baik untuk data yang linier separable. KNN memiliki akurasi lebih tinggi, yaitu 83%, dan unggul dalam menangkap pola kompleks, meskipun bisa lambat dengan dataset besar. SVM menghasilkan akurasi 79%, dilengkapi kemampuannya untuk menangani dataset berdimensi tinggi serta data non-linear melalui penggunaan kernel. Di sisi lain, Naive Bayes menunjukkan perbedaan signifikan antara Bernoulli NB (60%) yang lebih baik dibandingkan Gaussian NB (19%), ini mencerminkan ketidaksesuaian distribusi data dengan asumsi Gaussian. Decision Tree mencatat akurasi yang baik sebesar 86%, memanfaatkan kemampuan menangkap hubungan non-linear, walau berisiko melakukan overfitting tanpa regulasi. Random Forest menunjukkan performa luar biasa dengan akurasi 90%, berkat pendekatan ensemble yang mengurangi overfitting. XGBoost, dengan akurasi tertinggi 91%, dianggap sebagai salah satu algoritma boosting paling efektif, karena secara iteratif memperbaiki kesalahan model sebelumnya. Untuk model neural network, ANN mencatat akurasi 84% dan perlu tuning hyperparameter yang optimal, sementara RNN dan LSTM masing-masing memiliki akurasi 89% dan 90%, yang membuatnya ideal untuk data berurutan dan ketergantungan temporal seperti teks atau data waktu.

Kesimpulan utama dari evaluasi tanpa hyperparameter tuning ini menunjukkan bahwa XGBoost merupakan model terbaik dengan akurasi tertinggi sebesar 91%, diikuti oleh Random Forest dan LSTM yang keduanya memiliki akurasi 90%. Ketiga model ini unggul berkat kemampuan mereka dalam menangkap pola kompleks; XGBoost mengoptimalkan kesalahan secara iteratif dengan metode boosting, Random Forest menggunakan ensemble learning untuk mengurangi risiko overfitting, sedangkan LSTM dirancang khusus untuk menangani data dengan hubungan temporal. Di sisi lain, Gaussian Naive Bayes menunjukkan performa sangat rendah dengan akurasi hanya 19% karena asumsi distribusi normal pada fitur yang tidak sesuai dengan dataset ini, sementara Bernoulli Naive Bayes lebih baik dengan akurasi 60%, menunjukkan bahwa fitur yang digunakan lebih cocok dengan data biner. KNN (83%), ANN (84%), dan SVM (79%) memberikan hasil yang baik, tetapi tidak sekompetitif model ensemble atau jaringan saraf yang lebih kompleks. Faktor-faktor yang mempengaruhi perbedaan akurasi antara model termasuk sifat model—di mana XGBoost dan Random Forest menggunakan metode ensemble untuk meningkatkan akurasi dan generalisasi—serta kemampuan masing-masing model dalam menangkap pola dalam data. Misalnya. Secara keseluruhan, XGBoost terbukti menjadi pilihan terbaik untuk dataset ini, diikuti oleh Random Forest dan LSTM sebagai alternatif yang kompetitif, sementara model sederhana seperti Logistic Regression memberikan hasil yang memadai tetapi kalah dari model ensemble dan jaringan saraf yang lebih kompleks. Terakhir, Gaussian Naive Bayes gagal memberikan hasil yang baik dikarenakan ketidaksesuaian asumsi terhadap data.


| Model | Accuracy | Best Parameter |
| ------ | ------ | ------ |
| Logistic Regression | 0.79 | {'C': 0.01, 'penalty': 'l1', 'random_state': 0, 'solver': 'liblinear'} |
| KNN  | 0.93 | {'metric': 'manhattan', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'} |
| SVM | 0.88 | {'C': 1, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf', 'random_state': 0} |
| Naive Bayes (Bernoulli) | 0.76 | {'alpha': 0.1, 'binarize': 0.5} |
| Naive Bayes (Gaussian) | 0.58 | {'var_smoothing': 1e-06} |
| Decision Tree | 0.92 | {'criterion': 'entropy', 'max_depth': 30, 'min_samples_split': 2, 'random_state': 0, 'splitter': 'random'} |
| Random Forest | 0.9597 | {'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100, 'random_state': 0} |
| XGBOOST | 0.9584 | {'colsample_bytree': 0.6, 'eval_metric': 'error', 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 1.0} |
| ANN | 0.89 | {'batch_size': 25, 'epochs': 50, 'optimizer': 'adam'} |
| RNN | 0.39 | {'batch_size': 16, 'build_fn__dropout_rate': 0.3, 'build_fn__optimizer': 'adam', 'build_fn__units': 32, 'epochs': 30} |
| LSTM | 0.80 | {'batch_size': 16, 'build_fn__dropout_rate': 0.3, 'build_fn__optimizer': 'adam', 'build_fn__units': 32, 'epochs': 30} |

Tabel 3. Hasil Accuracy Machine Learning dan Deep learning dengan hyperparameter tuning

Tabel 3 merupakan hasil evaluasi metrik berupa accuracy terhadap model machine learning dan deep learning dengan menerapkan hyperparameter tuning. Analisis hasil evaluasi menunjukkan bahwa setelah melakukan hyperparameter tuning, Random Forest dan XGBoost mencatat akurasi tertinggi masing-masing sebesar 95.97% dan 95.84%. Random Forest, yang optimal pada parameter gini untuk pemisahan, kedalaman tak terbatas, dan penggunaan 100 pohon, sangat andal dalam menangani data non-linear dan noise berkat teknik ensemble yang digunakannya. Sementara itu, XGBoost unggul dengan parameter yang mengontrol kompleksitas pohon dan mencegah overfitting, berkat teknik boosting yang iteratif. Model KNN, dengan akurasi 93%, menunjukkan performa baik menggunakan 3 tetangga terdekat dan pengukuran jarak Manhattan, cocok untuk data dengan cluster yang terdefinisi jelas. Decision Tree juga berhasil dengan akurasi 92% berkat penggunaan entropi untuk pemisahan dan pembatasan kedalaman. Model dengan performa menengah seperti ANN (89%), SVM (88%), dan Logistic Regression (79%) menunjukkan kemampuannya di area tertentu, meskipun tidak sekompetitif model non-linear yang lebih kompleks. Di sisi lain, Naive Bayes (Bernoulli dan Gaussian) serta RNN (39%) dan LSTM (80%) menunjukkan performa lebih rendah, terutama karena keterbatasan asumsi distribusi dan kemampuan mereka dalam menangkap pola. Kesimpulannya, untuk akurasi tertinggi disarankan menggunakan Random Forest atau XGBoost; sementara Decision Tree menjadi pilihan baik jika interpretabilitas dibutuhkan, dan LSTM direkomendasikan untuk analisis pola waktu dan sekuensial.

## Kesimpulan
Tanpa penerapan hyperparameter tuning, model XGBoost menunjukkan performa terbaik dengan akurasi 91%, diikuti oleh Random Forest dan LSTM yang masing-masing mencapai 90%. Model-model ini unggul berkat kemampuannya dalam menangkap pola kompleks dan hubungan temporal dalam data, dengan XGBoost dan Random Forest menggunakan teknik ensemble untuk mengurangi overfitting, serta LSTM yang cocok untuk data berurutan. Sementara itu, model seperti Gaussian Naive Bayes dan KNN memiliki performa yang lebih rendah, dengan Naive Bayes bahkan mengalami penurunan drastis pada distribusi Gaussian. Model-model sederhana seperti Logistic Regression dan SVM memberikan hasil yang cukup baik tetapi kalah dibandingkan model ensemble atau jaringan saraf yang lebih kompleks.

Setelah penerapan hyperparameter tuning menggunakan GridSearch, Random Forest dan XGBoost mencatatkan akurasi tertinggi sebesar 95.97% dan 95.84%, menunjukkan bahwa tuning dapat secara signifikan meningkatkan performa model. Kedua model ini lebih stabil dan andal dalam menangani data non-linear dan noise berkat teknik ensemble dan boosting. Model KNN juga mengalami peningkatan, mencapai akurasi 93%, sementara model lainnya seperti ANN, SVM, dan Decision Tree menunjukkan hasil yang baik pada area tertentu. Namun, Naive Bayes dan RNN masih memiliki performa yang rendah, terutama karena asumsi distribusi yang tidak sesuai dengan data.

Secara keseluruhan, penerapan hyperparameter tuning dengan GridSearch memberikan hasil yang lebih optimal, dengan Random Forest dan XGBoost sebagai model unggulan untuk akurasi tertinggi, sedangkan Decision Tree dapat menjadi pilihan jika interpretabilitas dibutuhkan dan LSTM cocok untuk analisis data sekuensial atau temporal dalam prediksi stroke.

## Daftar Pustaka

1. None Indah Werdiningsih, Endah Purwanti, Iin Mardiyana, Arum Tiyas Handayani, Kharristantie Sekarlangit Suryadewi, Endang Nurjanah, Fildzah Akhlaqulkarimah, Naurah Hedy Pramiyas, & Almas, F. (2023). Analisis Prediksi Stroke Menggunakan Pendekatan Decision Tree dengan Seleksi Fitur dan Neural Network. Jurnal Sistem Cerdas, 6(3), 213–221. https://doi.org/10.37396/jsc.v6i3.310
2. Wulandari, E., & Arita Witanti. (2024). The CLASSIFICATION OF STROKE PREDICTION USING THE SUPPORT VECTOR MACHINE (SVM) METHOD. JATISI (Jurnal Teknik Informatika Dan Sistem Informasi), 11(3). https://doi.org/10.35957/jatisi.v11i3.8044
3. fedesoriano. (2021). Stroke Prediction Dataset. Kaggle.com. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
