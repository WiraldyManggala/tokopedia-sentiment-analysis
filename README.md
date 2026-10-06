Analisis Sentimen Ulasan Aplikasi Tokopedia

## Project Overview
Proyek ini berfokus pada analisis sentimen terhadap ulasan pengguna aplikasi Tokopedia di Google Play Store. Tujuan utama dari proyek ini adalah membangun model pemelajaran mendalam menggunakan arsitektur Bidirectional LSTM yang dikombinasikan dengan mekanisme Attention kustom. Model ini dirancang untuk mengklasifikasikan sentimen ulasan menjadi kategori positif, negatif, atau netral secara otomatis.

## Business Understanding

### Problem Statements
- Bagaimana cara memproses data teks ulasan pengguna berbahasa Indonesia agar dapat diekstraksi maknanya oleh model pemelajaran mesin?
- Bagaimana merancang arsitektur klasifikasi teks berbasis Deep Learning yang mampu memberikan tingkat akurasi tinggi pada data sentimen?

### Goals
- Mengimplementasikan prapemrosesan teks Natural Language Processing khusus untuk bahasa Indonesia.
- Membangun, melatih, dan mengevaluasi model klasifikasi sentimen menggunakan Bidirectional LSTM dan Attention layer.

### Solution Approach
- **Pengumpulan Data**: Melakukan scraping data ulasan langsung dari Google Play Store menggunakan library google-play-scraper.
- **Pemodelan**: Menggunakan arsitektur Bidirectional LSTM untuk menangkap urutan dan konteks kalimat dari dua arah, yang disempurnakan dengan Custom Attention Layer agar model dapat memfokuskan bobot komputasinya pada kata-kata yang paling menentukan sentimen.

## Data Understanding
Dataset yang digunakan diperoleh dari proses scraping aplikasi Tokopedia (com.tokopedia.tkpd) untuk wilayah dan bahasa Indonesia. Sebanyak 3000 baris data acak diambil untuk keperluan eksperimen dan disimpan dalam bentuk CSV (google_play_tokopedia_3000.csv).

Fitur utama dalam dataset:
- `content`: Teks ulasan mentah dari pengguna aplikasi.
- `score`: Skor bintang aplikasi dari angka 1 hingga 5.
- `sentiment`: Label target klasifikasi yang dibuat berdasarkan score, dengan ketentuan score >= 4 sebagai positive, score <= 2 sebagai negative, dan score 3 sebagai neutral.

## Data Preparation
Tahapan prapemrosesan teks dilakukan untuk membersihkan dan menstandarisasi data sebelum masuk ke tahap pemodelan:
- **Text Cleaning**: Meliputi proses lowercasing, penghapusan karakter non-alfabet menggunakan regex, penghapusan stopwords bahasa Indonesia menggunakan NLTK, serta proses stemming menggunakan Sastrawi.
- **Label Encoding**: Transformasi label kelas menjadi bentuk numerik menggunakan LabelEncoder, yang selanjutnya diubah menjadi representasi one-hot encoding dengan fungsi to_categorical.
- **Tokenisasi dan Padding**: Konversi teks ke dalam bentuk sekuens angka menggunakan Tokenizer (dengan batas 10.000 kata unik). Sekuens kemudian disamakan panjang maksimalnya menjadi 100 menggunakan pad_sequences.
- **Pemisahan Dataset**: Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan rasio pemisahan bertingkat pada target kelas.

## Modeling
Model dibangun secara fungsional menggunakan TensorFlow/Keras API dengan rincian arsitektur sebagai berikut:
1. **Input Layer**: Menerima fitur sekuens panjang maksimal 100 dimensi.
2. **Embedding Layer**: Mengubah input token menjadi vektor dimensi padat berukuran 128.
3. **Bidirectional LSTM**: Terdiri dari 64 unit LSTM yang mengembalikan sekuens utuh untuk diproses pada lapisan berikutnya.
4. **Custom Attention Layer**: Lapisan mekanika atensi yang menghitung bobot spesifik pada fitur tersembunyi dari LSTM.
5. **Dropout Layer**: Diterapkan dengan nilai rate sebesar 0.5 untuk regulasi dan mencegah overfitting.
6. **Output Layer**: Menggunakan fungsi aktivasi softmax untuk memprediksi probabilitas pada 3 kelas target.

Model ini dikompilasi menggunakan optimizer adam dan metrik fungsi kerugian categorical_crossentropy.

## Evaluation
Model dilatih selama 10 iterasi (epoch) menggunakan batch berukuran 32, dan mengalokasikan 20% data latih sebagai himpunan validasi.

Hasil pengujian pada data uji menunjukkan performa klasifikasi yang cukup relevan:
- **Akurasi**: Model berhasil mencapai tingkat akurasi sekitar 86.67% pada himpunan data uji.
- Terdapat fungsi bawaan untuk memvisualisasikan Confusion Matrix yang menampilkan perbandingan prediksi versus kebenaran aktual, serta laporan Classification Report untuk nilai metrik Precision, Recall, dan F1-Score.

## Setup Environment
Persiapkan dependensi lingkungan kerja Anda dengan menjalankan perintah berikut melalui terminal atau command prompt:
```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib tensorflow sastrawi google-play-scraper
