Analisis Sentimen Ulasan Aplikasi Tokopedia

## Project Overview
Proyek ini berfokus pada analisis sentimen terhadap ulasan pengguna aplikasi Tokopedia di Google Play Store[cite: 13]. Tujuan utama dari proyek ini adalah membangun model pemelajaran mendalam (Deep Learning) menggunakan arsitektur Bidirectional LSTM yang dikombinasikan dengan mekanisme Attention kustom[cite: 14]. Model ini dirancang untuk mengklasifikasikan sentimen ulasan menjadi kategori positif, negatif, atau netral secara otomatis[cite: 13, 14].

## Business Understanding

### Problem Statements
- Bagaimana cara memproses data teks ulasan pengguna berbahasa Indonesia agar dapat diekstraksi maknanya oleh model pemelajaran mesin?
- Bagaimana merancang arsitektur klasifikasi teks berbasis Deep Learning yang mampu memberikan tingkat akurasi tinggi pada data sentimen?

### Goals
- Mengimplementasikan prapemrosesan teks Natural Language Processing (NLP) khusus untuk bahasa Indonesia.
- Membangun, melatih, dan mengevaluasi model klasifikasi sentimen menggunakan Bidirectional LSTM dan Attention layer.

### Solution Approach
- **Pengumpulan Data**: Melakukan *scraping* data ulasan langsung dari Google Play Store menggunakan library `google-play-scraper`[cite: 13].
- **Pemodelan**: Menggunakan arsitektur Bidirectional LSTM untuk menangkap urutan dan konteks kalimat dari dua arah, yang disempurnakan dengan *Custom Attention Layer* agar model dapat memfokuskan bobot komputasinya pada kata-kata yang paling menentukan sentimen[cite: 14].

## Data Understanding
Dataset yang digunakan diperoleh dari proses *scraping* aplikasi Tokopedia (`com.tokopedia.tkpd`) untuk wilayah dan bahasa Indonesia[cite: 13]. Sebanyak 3000 baris data acak diambil untuk keperluan eksperimen dan disimpan dalam bentuk CSV (`google_play_tokopedia_3000.csv`)[cite: 13].

Fitur utama dalam dataset:
- `content`: Teks ulasan mentah dari pengguna aplikasi[cite: 13].
- `score`: Skor bintang aplikasi dari angka 1 hingga 5[cite: 13].
- `sentiment`: Label target klasifikasi yang dibuat berdasarkan `score`, dengan ketentuan skor >= 4 sebagai `positive`, skor <= 2 sebagai `negative`, dan skor 3 sebagai `neutral`[cite: 13].

## Data Preparation
Tahapan prapemrosesan teks dilakukan untuk membersihkan dan menstandarisasi data sebelum masuk ke tahap pemodelan[cite: 14]:
- **Text Cleaning**: Meliputi proses lowercasing, penghapusan karakter non-alfabet menggunakan regex, penghapusan *stopwords* bahasa Indonesia menggunakan NLTK, serta proses *stemming* menggunakan Sastrawi[cite: 14].
- **Label Encoding**: Transformasi label kelas menjadi bentuk numerik menggunakan `LabelEncoder`, yang selanjutnya diubah menjadi representasi *one-hot encoding* dengan fungsi `to_categorical`[cite: 14].
- **Tokenisasi dan Padding**: Konversi teks ke dalam bentuk sekuens angka menggunakan `Tokenizer` (dengan batas 10.000 kata unik). Sekuens kemudian disamakan panjang maksimalnya menjadi 100 menggunakan `pad_sequences`[cite: 14].
- **Pemisahan Dataset**: Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan rasio pemisahan bertingkat (*stratified*) pada target kelas[cite: 14].

## Modeling
Model dibangun secara fungsional menggunakan TensorFlow/Keras API dengan rincian arsitektur sebagai berikut[cite: 14]:
1. **Input Layer**: Menerima fitur sekuens panjang maksimal 100 dimensi[cite: 14].
2. **Embedding Layer**: Mengubah input token menjadi vektor dimensi padat berukuran 128[cite: 14].
3. **Bidirectional LSTM**: Terdiri dari 64 unit LSTM yang mengembalikan sekuens utuh untuk diproses pada lapisan berikutnya[cite: 14].
4. **Custom Attention Layer**: Lapisan mekanika atensi yang menghitung bobot spesifik pada fitur tersembunyi (*hidden features*) dari LSTM[cite: 14].
5. **Dropout Layer**: Diterapkan dengan nilai *rate* sebesar 0.5 untuk regulasi dan mencegah *overfitting*[cite: 14].
6. **Output Layer**: Menggunakan fungsi aktivasi *softmax* untuk memprediksi probabilitas pada 3 kelas target[cite: 14].

Model ini dikompilasi menggunakan optimizer `adam` dan metrik fungsi kerugian `categorical_crossentropy`[cite: 14].

## Evaluation
Model dilatih selama 10 iterasi (epoch) menggunakan batch berukuran 32, dan mengalokasikan 20% data latih sebagai himpunan validasi (*validation split*)[cite: 14].

Hasil pengujian pada data uji menunjukkan performa klasifikasi yang cukup relevan:
- **Akurasi**: Model berhasil mencapai tingkat akurasi sekitar 86.67% pada himpunan data uji[cite: 14].
- Terdapat fungsi bawaan untuk memvisualisasikan *Confusion Matrix* yang menampilkan perbandingan prediksi versus kebenaran aktual, serta laporan *Classification Report* untuk nilai metrik *Precision*, *Recall*, dan *F1-Score*[cite: 14].

## Setup Environment
Persiapkan dependensi lingkungan kerja Anda dengan menjalankan perintah berikut melalui terminal atau *command prompt*[cite: 13, 14]:
```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib tensorflow sastrawi google-play-scraper
