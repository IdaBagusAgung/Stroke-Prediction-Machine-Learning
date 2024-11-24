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







### EDA - Multivariate Analysis








## Daftar Pustaka

1. None Indah Werdiningsih, Endah Purwanti, Iin Mardiyana, Arum Tiyas Handayani, Kharristantie Sekarlangit Suryadewi, Endang Nurjanah, Fildzah Akhlaqulkarimah, Naurah Hedy Pramiyas, & Almas, F. (2023). Analisis Prediksi Stroke Menggunakan Pendekatan Decision Tree dengan Seleksi Fitur dan Neural Network. Jurnal Sistem Cerdas, 6(3), 213–221. https://doi.org/10.37396/jsc.v6i3.310

‌2. Wulandari, E., & Arita Witanti. (2024). The CLASSIFICATION OF STROKE PREDICTION USING THE SUPPORT VECTOR MACHINE (SVM) METHOD. JATISI (Jurnal Teknik Informatika Dan Sistem Informasi), 11(3). https://doi.org/10.35957/jatisi.v11i3.8044

‌3. fedesoriano. (2021). Stroke Prediction Dataset. Kaggle.com. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

‌4. 
