# Laporan Proyek Sistem Rekomendasi Smartphone

Proyek Machine Learning untuk mengembangkan sistem rekomendasi smartphone menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering.

## Project Overview

### Latar Belakang

Dalam era digital saat ini, pasar smartphone berkembang sangat pesat dengan ribuan model yang tersedia dari berbagai brand. Konsumen sering mengalami kesulitan dalam memilih smartphone yang sesuai dengan kebutuhan dan preferensi mereka karena banyaknya pilihan yang tersedia. Fenomena "choice overload" ini menciptakan kebutuhan akan sistem rekomendasi yang dapat membantu konsumen menemukan smartphone yang tepat.

E-commerce dan platform review teknologi membutuhkan sistem rekomendasi yang efektif untuk meningkatkan kepuasan pelanggan dan conversion rate. Menurut penelitian dari MIT Sloan, sistem rekomendasi yang baik dapat meningkatkan penjualan hingga 35% dan engagement pengguna hingga 60%.


### Mengapa Proyek Ini Penting?

+ Membantu pengguna menemukan smartphone yang sesuai dengan preferensi tanpa harus melalui ratusan pilihan
+ Mengurangi waktu yang dibutuhkan konsumen untuk research dan membandingkan spesifikasi
+ Meningkatkan conversion rate dan customer satisfaction pada platform e-commerce
+ Memberikan rekomendasi yang disesuaikan dengan riwayat rating dan preferensi individual pengguna

### Referensi

+ Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer. https://www.researchgate.net/publication/227268858_Recommender_Systems_Handbook
+ Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
https://ieeexplore.ieee.org/document/5197422

## Business Understanding

### Problem Statements

Berdasarkan analisis kebutuhan bisnis dan pengalaman pengguna, permasalahan utama yang ingin diselesaikan adalah:

+ Konsumen kesulitan memilih smartphone yang tepat dari ratusan pilihan yang tersedia di pasar
+ Platform e-commerce belum memberikan rekomendasi yang dipersonalisasi berdasarkan preferensi individual pengguna
+ Terlalu banyak informasi spesifikasi teknis yang membingungkan konsumen awam
+ Tingkat interaksi pengguna dengan platform rendah karena sulitnya menemukan produk yang relevan

### Goals

Tujuan dari proyek sistem rekomendasi smartphone ini adalah:

+ Mengembangkan sistem rekomendasi hybrid yang menggabungkan Content-Based Filtering dan Collaborative Filtering
+ Memberikan rekomendasi personal berdasarkan spesifikasi smartphone dan pola rating pengguna
+ Meningkatkan akurasi prediksi dengan target RMSE < 1.0 untuk Collaborative Filtering
+ Mencapai diversity score minimal 0.5 untuk memastikan variasi rekomendasi yang cukup
+ Menghasilkan top-N recommendations yang relevan dan mudah dipahami pengguna

### Solution Approach

1. Content-Based Filtering

Merekomendasikan smartphone berdasarkan kesamaan spesifikasi teknis dengan smartphone yang disukai pengguna sebelumnya.

+ Fitur yang digunakan:

    + Brand smartphone
    + Operating System (Android/iOS)
    + Performance score
    + Kamera utama (megapixel)
    + Kapasitas baterai
    + Ukuran layar
    + Berat smartphone

2. Collaborative Filtering

Menggunakan Neural Network dengan embedding layers untuk memprediksi rating smartphone berdasarkan pola preferensi pengguna dan item similarity.

+ Arsitektur Model:

    + User dan Item Embedding layers (50 dimensi)
    + Dense layers dengan dropout untuk regularization
    + Output layer untuk prediksi rating (1-5)

## Data Understanding

### Informasi Dataset

+ Sumber Data: Dataset Smartphone dari Kaggle https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations/data
+ Format Data: 3 file CSV terpisah
  + Data smartphone (cellphones data/data.csv):  33 data ponsel, 14 atribut
  + Rating data (cellphones ratings/rating.csv): 990 data penilaian, 3 kolom (user_id, cellphone_id, rating)
  + User data (cellphones users/user.csv): 99 pengguna, 4 kolom (user_id, age, gender, occupation).

### Kondisi Data

Setelah melakukan pemeriksaan awal, kondisi dataset adalah sebagai berikut:

+ Missing Values: Ditemukan beberapa baris dengan nilai kosong yang perlu dibersihkan
+ Duplicate Records: Ada duplikasi pada cellphone_id yang perlu dihapus
+ Anomali Data: Rating dengan nilai 18 (outlier) yang perlu difilter
+ Data Quality Issues: Typo pada kolom occupation ('healthare' → 'healthcare', 'it' → 'information technology')

### Variabel/Fitur dalam Dataset

+ cellphone data.csv/data.csv

![image](https://github.com/user-attachments/assets/992fe606-2153-42f9-bc07-2985c5570897)

+ cellphone ratings.csv/rating.csv

![image](https://github.com/user-attachments/assets/8df6a6c4-8061-42ca-8392-fbf852443f5b)

+ cellphone users.csv/user.csv

![image](https://github.com/user-attachments/assets/ba7f6225-3e9f-4d13-a555-9f0a1930b583)


### Exploratory Data Analysis (EDA)

1. Data

+ Distribusi Brand Smartphone

![image](https://github.com/user-attachments/assets/a98a353e-d30a-4c0e-8e50-8511012ca071)

Analisis menunjukkan bahwa dataset memiliki distribusi brand yang beragam dengan di dominasi oleh Samsung dengan 8 model, diikuti Apple dengan 6, sementara merek lain memiliki jumlah model yang lebih sedikit dan bervariasi.

+ Distribusi Sistem Operasi

![image](https://github.com/user-attachments/assets/9e8cc576-9379-4d46-b861-a40b941d19a4)

Distribusi sistem operasi didominasi oleh Android dengan 27 model, sementara iOS hanya memiliki 6 model dalam dataset.

+ Distribusi Kapasitas Memori Internal

![image](https://github.com/user-attachments/assets/8eb3e507-47c9-474a-bcee-a745e70d983f)

Kapasitas memori internal yang paling umum adalah 128GB dengan 20 model, diikuti oleh 256GB, 64GB, 32GB, dan hanya satu model dengan 512GB.

+ Distribusi Kapasitas RAM

![image](https://github.com/user-attachments/assets/b3986cc5-221d-44bc-97df-9c74ca807d23)

Kapasitas RAM paling banyak tersedia adalah 8GB dengan 13 model, diikuti oleh 4GB dan 6GB masing-masing 6 model, serta 3GB dan 12GB yang lebih sedikit.

+  Kategorisasi Performa

![image](https://github.com/user-attachments/assets/e8c8ad13-7e45-4e4f-9dda-2c8a549dc6f8)

Terdapat 23 smartphone dengan performa tinggi (skor > 5) dan 10 smartphone dengan performa rendah (skor ≤ 5) dalam dataset.

+ Analisis Kamera Utama

![image](https://github.com/user-attachments/assets/d31ab2f4-15ba-4c5d-83b1-385f02c47647)

Resolusi kamera utama yang paling umum adalah 50MP dengan 13 model, diikuti 12MP dengan 10 model, serta beberapa model dengan resolusi lebih tinggi seperti 64MP, 48MP, dan 108MP.

+ Analisis Kapasitas Baterai

![image](https://github.com/user-attachments/assets/ec6e3c60-731f-4bc4-9d77-96956380c33e)

Kapasitas baterai 5000mAh paling umum dengan 11 model, sementara kapasitas lainnya bervariasi dari 2018mAh hingga 5003mAh dengan jumlah model lebih sedikit.

+ Analisis Ukuran Layar

![image](https://github.com/user-attachments/assets/b2720a1d-69ed-490c-bba0-899dfd2b9d44)

Ukuran layar 6.7 inci paling umum dengan 8 model, diikuti ukuran 6.5 inci dan 6.1 inci, serta variasi ukuran layar lain yang lebih kecil jumlahnya.

+ Analisis Berat Smartphone

![image](https://github.com/user-attachments/assets/34779463-8e9d-444e-b512-71a0e8779a17)

Berat smartphone bervariasi dalam 27 kategori berbeda, dengan berat 204 gram paling sering muncul sebanyak 5 model.

+ Analisis Tahun Rilis

![image](https://github.com/user-attachments/assets/79f59dc3-3500-449b-9bee-df59c52972d4)

Smartphone dalam dataset dirilis terutama pada tahun 2021 dan 2022 dengan jumlah yang sama, sementara hanya satu model yang dirilis pada 2018.

2. Rating

+ Aktivitas Penguna

![image](https://github.com/user-attachments/assets/3babe50d-c838-4347-bcdb-97de260f0ebb)

Setiap user dalam dataset rating memberikan tepat 10 ulasan, menunjukkan distribusi aktivitas review yang seragam antar pengguna.

+ Popularitas Smartphone

![image](https://github.com/user-attachments/assets/086bfd55-5f29-4a67-81c7-65866b457831)

Jumlah review per smartphone berkisar antara 20 hingga 41, menunjukkan variasi popularitas antar model.

+ Distribusi Rating

![image](https://github.com/user-attachments/assets/2030b384-cc0e-4376-9d34-d57721caba39)

Sebagian besar pengguna memberikan rating tinggi (7–10), dengan rating 8 sebagai yang paling sering muncul.

3. User

+ Distribusi Rating

![image](https://github.com/user-attachments/assets/b30f2178-5caa-44e0-ba58-e41d11a7489a)

Distribusi usia pengguna beragam, dengan konsentrasi tertinggi pada usia 25 tahun dan rentang usia pengguna dari 21 hingga 61 tahun.

+ Distribusi Gender

![image](https://github.com/user-attachments/assets/f5a75de9-f3d2-4ee9-bbf1-8fc36f0ed814)

Mayoritas pengguna adalah laki-laki (50) dan perempuan (46), dengan (3) data yang belum terisi pada kategori gender.

+ Distribusi Pekerjaan

Pengguna memiliki beragam pekerjaan dengan 45 variasi berbeda, paling banyak berprofesi sebagai manager, information technology, dan IT.

## Data Preparation

1. Integrasi Data

Proses Penggabungan Dataset

Preprocessing menggabungkan tiga dataset utama:

+ Dataset Rating (rating_df) - Berisi rating pengguna untuk smartphone
  + Menyimpan hubungan antara user_id dan cellphone_id
  + Mencatat nilai rating yang diberikan pengguna
+ Dataset Smartphone (data_df) - Berisi spesifikasi dan detail smartphone
  + Informasi teknis seperti brand, model, operating system
  + Spesifikasi hardware dan fitur smartphone
+ Dataset User (user_df) - Berisi informasi demografis pengguna
  + Data profil pengguna seperti gender, occupation, age
  + Informasi preferensi dan karakteristik pengguna
 
![image](https://github.com/user-attachments/assets/0c95b032-8815-44a6-a801-82e59489e2ae)

Hasil: Dataset terintegrasi memberikan informasi lengkap yang menghubungkan spesifikasi smartphone, profil pengguna, dan rating yang diberikan.

2. Penilaian Kualitas Data

+ Analisis Missing Value

  + Kondisi Awal: Hanya kolom occupation yang memiliki missing value (10 data kosong)
  + Sumber Masalah: Semua missing value berasal dari user_id 53
  + Masalah Tambahan: User yang sama memiliki entry gender tidak valid ("-Select Gender-")

Penjelasan: Missing value pada occupation menunjukkan data tidak lengkap dari satu pengguna tertentu, yang dapat mempengaruhi kualitas analisis demografis.

+ Operasi Pembersihan Data
  + Penanganan Missing Value
  
    ![image](https://github.com/user-attachments/assets/72d4e458-9a0b-47c3-861c-1f0dee4d20ce)

  Hasil: Eliminasi lengkap missing value di seluruh kolom.
  
  Penjelasan: Penghapusan baris dilakukan karena jumlah missing value relatif kecil (10 dari total data) dan berasal dari satu user yang datanya tidak konsisten, sehingga penghapusan tidak signifikan mengurangi volume data yang berguna.

  + Penghapusan Anomali
    
    ![image](https://github.com/user-attachments/assets/9b071624-1815-4f8c-8d39-ac818b81f8e0)
    
    Alasan: Nilai rating 18 diidentifikasi sebagai outlier yang tidak konsisten dengan skala rating yang diharapkan (kemungkinan kesalahan input atau data corrupt).
    
    Penjelasan: Rating 18 kemungkinan adalah kesalahan input atau data corrupt yang dapat mengganggu akurasi model rekomendasi karena berada di luar rentang rating yang valid (1-10) yang diamati dalam dataset.

  + Standardisasi Data
 
    ![image](https://github.com/user-attachments/assets/f23c2511-8b86-4c52-91f3-0cc32a540985)

    Normalisasi Field Occupation:
    
    + Konversi semua entry ke lowercase untuk konsistensi
    + Perbaikan typo: 'healthare' → 'healthcare'
    + Perluasan singkatan: 'it' → 'information technology'
      
    Penjelasan: Standardisasi occupation penting untuk memastikan kategorisasi yang konsisten dalam analisis demografis dan clustering pengguna, serta untuk meminimalisir variasi data yang tidak perlu akibat kesalahan penulisan.

3. Persiapan Fitur per Pendekatan

+ Content-Based Filtering Data Preparation

    + Pembuatan DataFrame Fitur Ponsel
      
      Untuk Content-Based Filtering, duplikat berdasarkan cellphone_id dihapus dari processed_data untuk memastikan setiap ponsel pintar unik terwakili sekali. Kolom-kolom kunci yang relevan untuk rekomendasi berbasis konten (cellphone_id, brand, model, operating system, performance, main camera, battery size, screen size, weight, price) diekstrak untuk membentuk phone_dataset. Dataset ini akan menjadi dasar untuk ekstraksi fitur teks dan numerik.
      
    + Penggabungan Fitur Teks dan Numerik
      
      Fitur-fitur seperti brand, model, operating system, performance, main camera, battery size, screen size, weight, dan price dari phone_dataset digabungkan menjadi satu string content_profile untuk setiap ponsel. Fitur numerik (battery size, screen size, weight, price) dinormalisasi ke dalam rentang 0-1 untuk memastikan kontribusi yang seimbang dalam perhitungan kesamaan.
      
    + Vektorisasi TF-IDF
      
      Teknik TF-IDF (Term Frequency-Inverse Document Frequency) diterapkan pada kolom content_profile yang telah digabungkan. TfidfVectorizer dari sklearn.feature_extraction.text digunakan untuk mengubah teks menjadi matriks fitur numerik.

        + Term Frequency (TF): Mengukur seberapa sering sebuah kata muncul dalam sebuah dokumen (dalam hal ini, profil konten ponsel).
        + Inverse Document Frequency (IDF): Memberikan bobot lebih rendah pada kata-kata yang sangat umum di seluruh dokumen (profil konten) dan bobot lebih tinggi pada kata-kata yang lebih jarang.
          
    Hasilnya adalah tfidf_matrix, sebuah matriks yang menunjukkan pentingnya setiap kata dalam mendeskripsikan setiap ponsel. Proses ini esensial untuk mengubah data teks menjadi format numerik yang dapat diproses oleh algoritma kesamaan.

+ Collaborative Filtering Data Preparation

    Untuk model Collaborative Filtering, data perlu disiapkan dalam format yang sesuai untuk melatih model jaringan saraf:
    
    + Pembuatan Matriks Pengguna-Item
    
    Dari processed_data, sebuah matriks user-item dibuat di mana baris mewakili pengguna (user_id), kolom mewakili ponsel pintar (cellphone_id), dan nilai adalah penilaian (rating). Nilai yang hilang (yang berarti pengguna belum menilai ponsel tersebut) diisi dengan 0. Matriks ini menjadi representasi utama interaksi pengguna-item.
    
    + Pengkodean ID untuk Embedding
      
    user_id dan cellphone_id dikodekan ulang menjadi indeks numerik berurutan. Ini diperlukan karena lapisan embedding dalam jaringan saraf membutuhkan input integer yang berurutan. Pemetaan dari ID asli ke ID yang dikodekan disimpan untuk rekonstruksi nanti.
    
    + Pemisahan Data (Data Split)
    
    Data yang telah diproses (processed_data) kemudian dibagi menjadi set pelatihan dan pengujian menggunakan train_test_split dari sklearn.model_selection.
    
        + Rasio Pemisahan: Data dibagi dengan rasio 80% untuk pelatihan dan 20% untuk pengujian (test_size=0.2).
        + Stratifikasi: Pemisahan dilakukan secara acak dengan random_state yang tetap untuk memastikan reproduktibilitas.
        
    Pemisahan ini memungkinkan evaluasi model yang adil pada data yang belum pernah dilihat selama pelatihan, memastikan bahwa model tidak hanya menghafal data pelatihan (overfitting).

## Modeling

Proyek ini mengimplementasikan dua pendekatan sistem rekomendasi yang berbeda: Content-Based Filtering dan Collaborative Filtering.

1. Content-Based Filtering

+ Pendekatan: 

    + Kesamaan Kosinus (Cosine Similarity): cosine_similarity diterapkan pada tfidf_matrix yang telah dihasilkan pada tahap persiapan data untuk menghitung skor kesamaan antara setiap pasang ponsel pintar. Skor yang lebih tinggi menunjukkan kesamaan fitur yang lebih besar.
      
    + Generasi Rekomendasi: Diberikan phone_id, sistem mengidentifikasi indeksnya dalam matriks kesamaan, mengambil skor kesamaan dengan semua ponsel lain, mengurutkannya dalam urutan menurun, dan mengembalikan N ponsel paling mirip (tidak termasuk dirinya sendiri).

+ Kelebihan:
  
    + Tidak Ada Cold-Start untuk Pengguna Baru: Dapat merekomendasikan item kepada pengguna baru asalkan ada informasi tentang item itu sendiri.
    + Transparansi: Rekomendasi dapat dijelaskan berdasarkan fitur item, memungkinkan pengguna memahami mengapa suatu item direkomendasikan.
    + Menangani Item Niche: Dapat merekomendasikan item niche jika fitur-fiturnya cocok dengan preferensi pengguna yang spesifik.
      
+ Kekurangan:
  
    + Serendipitas Terbatas: Cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, membatasi penemuan item baru atau beragam.
    + Biaya Rekayasa Fitur (Feature Engineering): Membutuhkan fitur item yang terperinci dan terstruktur dengan baik, yang bisa sulit didapatkan dan diproses.
      
2. Collaborative Filtering

+ Pendekatan:

    + Arsitektur Model Jaringan Saraf (NCF): Sebuah model Neural Collaborative Filtering dibangun menggunakan TensorFlow/Keras.
      
      + Embedding Layers: Lapisan embedding terpisah dibuat untuk pengguna dan item. Ukuran embedding adalah 50 dimensi untuk pengguna dan 50 dimensi untuk item. Lapisan ini mengubah ID pengguna dan item yang sparse menjadi vektor representasi dense yang dapat menangkap pola hubungan.
      + Perataan (Flatten): Vektor embedding dari pengguna dan item diratakan menggunakan layers.Flatten().
      + Konkatenasi: Vektor embedding pengguna dan item yang diratakan kemudian digabungkan (layers.concatenate) menjadi satu vektor fitur.
      + Hidden Layers: Beberapa lapisan dense ditambahkan untuk mempelajari interaksi kompleks antara pengguna dan item.
        + Lapisan Dense pertama memiliki 128 unit dan aktivasi ReLU (activation='relu').
        + Lapisan Dropout dengan rate 0.2 diterapkan setelah lapisan pertama untuk mencegah overfitting.
        + Lapisan Dense kedua memiliki 64 unit dan aktivasi ReLU (activation='relu').
        + Lapisan Dropout dengan rate 0.5 diterapkan setelah lapisan kedua.
        + Lapisan Dense ketiga memiliki 32 unit dan aktivasi ReLU (activation='relu').
      + Output Layer: Lapisan dense terakhir memiliki 1 unit dengan aktivasi linear (activation='linear') untuk memprediksi nilai penilaian.
        
    + Kompilasi Model: Model dikompilasi dengan:
 
      + Fungsi Loss: Mean Squared Error (MSE), yang mengukur rata-rata kuadrat selisih antara nilai prediksi dan aktual.
      + Optimizer: Adam, sebuah algoritma optimasi adaptif yang efisien.
      + Metrik: Mean Absolute Error (MAE), yang mengukur rata-rata selisih absolut antara nilai prediksi dan aktual, memberikan metrik yang lebih mudah diinterpretasikan dibandingkan MSE.
        
    + Pelatihan Model: Model dilatih menggunakan data pelatihan dengan parameter:
 
      + Jumlah Epoch: 30
      + Batch Size: 128
      + Early Stopping: Sebuah callback EarlyStopping digunakan untuk memantau loss validasi (val_loss) dan menghentikan pelatihan jika loss tidak membaik selama 3 epoch berturut-turut (patience=5). Ini mencegah overfitting dan menghemat waktu komputasi.

+ Kelebihan:
  
    + Serendipitas: Dapat menemukan item baru yang mungkin disukai pengguna meskipun mereka tidak berbagi kesamaan fitur yang jelas dengan item yang disukai sebelumnya, dengan memanfaatkan preferensi pengguna yang serupa.
    + Tidak Ada Rekayasa Fitur: Tidak memerlukan fitur item eksplisit; hanya mengandalkan data interaksi pengguna-item (penilaian), sehingga prosesnya lebih otomatis.
      
+ Kekurangan:
  
    + Cold-Start Problem: Kesulitan dengan pengguna baru (tidak memiliki riwayat penilaian) atau item baru (tidak memiliki penilaian), karena model tidak dapat membentuk kesamaan atau memprediksi preferensi tanpa data interaksi.
    + Sparsitas: Kinerja dapat menurun secara signifikan jika matriks interaksi pengguna-item sangat sparse (sangat sedikit penilaian dibandingkan dengan kemungkinan interaksi), karena model kesulitan menemukan pola yang cukup.

## Evaluasi

Kinerja kedua model rekomendasi dinilai menggunakan metrik yang relevan.

1. Evaluasi Content-Based Filtering

+ Metrik yang Digunakan:
  
    + Presisi (Precision): Mengukur proporsi item yang direkomendasikan yang benar-benar relevan bagi pengguna. Presisi yang tinggi menunjukkan bahwa model tidak terlalu banyak memberikan rekomendasi yang salah.
    + Recall: Mengukur proporsi item relevan yang berhasil direkomendasikan dari semua item relevan yang ada. Recall yang tinggi menunjukkan bahwa model menemukan sebagian besar item yang relevan.
    + F1-Score: Rata-rata harmonik dari Presisi dan Recall, memberikan skor tunggal yang menyeimbangkan kedua metrik. Ini berguna ketika ada ketidakseimbangan antara pentingnya Presisi dan Recall.
    + Skor Keberagaman (Diversity Score): Mengukur variasi merek dalam rekomendasi, menunjukkan seberapa luas rekomendasi tersebut. Ini dihitung sebagai proporsi merek unik dalam daftar rekomendasi.
    + Cakupan Katalog (Catalog Coverage): Mengukur proporsi semua item yang tersedia dalam katalog yang dapat direkomendasikan oleh sistem.

+ Hasil Evaluasi:
  
    + Precision: 0.2200
    + Recall: 0.1876
    + F1-Score: 0.2025
    + Diversity Score: 0.1200
    + Coverage: 0.8182
  
+ Statistik Kesamaan:

    + Rata-rata (Mean): 0.2041
    + Std: 0.3031
    + Min: 0.0000
    + Max: 1.0000
    + Median: 0.1352

2. Evaluasi Collaborative Filtering

+ Metrik yang Digunakan:
  
    + Root Mean Squared Error (RMSE): Mengukur akar kuadrat dari rata-rata kuadrat selisih antara penilaian yang diprediksi dan aktual. RMSE yang lebih rendah menunjukkan akurasi prediksi yang lebih baik, karena memberikan bobot lebih pada kesalahan besar.
    + Mean Absolute Error (MAE): Mengukur rata-rata selisih absolut antara penilaian yang diprediksi dan aktual. MAE yang lebih rendah menunjukkan akurasi prediksi yang lebih baik dan lebih mudah diinterpretasikan karena tidak membebani kesalahan besar secara berlebihan.
    + Mean Squared Error (MSE): Mengukur rata-rata kuadrat selisih antara penilaian yang diprediksi dan aktual. Metrik ini digunakan sebagai fungsi loss selama pelatihan model.
    + Precision@K: Mengukur proporsi item relevan di antara K rekomendasi teratas yang diberikan kepada pengguna. Digunakan untuk menilai seberapa akurat model dalam merekomendasikan item yang relevan di daftar teratas.
    + Recall@K: Mengukur proporsi item relevan yang ditemukan dalam K rekomendasi teratas dibandingkan dengan semua item relevan yang sebenarnya. Digunakan untuk menilai seberapa banyak item relevan yang berhasil ditangkap oleh model.
    + F1-Score@K: Rata-rata harmonik dari Precision@K dan Recall@K, memberikan keseimbangan antara kedua metrik tersebut, terutama ketika keduanya sama pentingnya.
      
+ Evaluasi Hasil:

    + RMSE: 2.0059
    + MAE: 1.4614
    + MSE: 4.0237
    + Precision@5: 0.0571
    + Recall@5: 0.0952
    + F1-Score@5: 0.0714

+ Plot Riwayat Pelatihan:

![image](https://github.com/user-attachments/assets/e194925a-6f9d-4078-a2b1-81826e333aed)

Evaluasi collaborative filtering menunjukkan performa cukup baik dengan nilai RMSE sebesar 2.0059, MAE 1.4614, dan MSE 4.0237. Grafik training menunjukkan penurunan loss dan MAE yang stabil, mengindikasikan model belajar dengan baik. Namun, performa metrik ranking masih rendah dengan Precision@5 sebesar 0.0571, Recall@5 sebesar 0.0952, dan F1-Score@5 sebesar 0.0714, menandakan bahwa kualitas rekomendasi top-5 masih perlu ditingkatkan.

+ Perbandingan Hasil Evaluasi

![image](https://github.com/user-attachments/assets/617579f3-d5bb-4666-988b-a0bed4f5ab03)

Model Content-Based Filtering memiliki performa lebih baik dibandingkan Collaborative Filtering, dengan nilai precision, recall, F1-score, dan coverage yang lebih tinggi. Oleh karena itu, Content-Based lebih efektif untuk sistem rekomendasi dalam kasus ini.
