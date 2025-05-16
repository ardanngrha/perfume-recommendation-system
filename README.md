# Laporan Proyek Machine Learning - Ardana Aldhizuma Nugraha

## Project Overview

Parfum telah menjadi bagian penting dalam kehidupan manusia sejak ribuan tahun lalu, bukan hanya sebagai penyegar bau, tetapi juga sebagai bentuk ekspresi identitas pribadi. Di era modern, industri parfum telah berkembang pesat dengan ribuan produk yang tersedia di pasaran, menyulitkan konsumen untuk menemukan aroma yang sesuai dengan preferensi mereka. Menurut penelitian yang dilakukan oleh Grand View Research, pasar global parfum mencapai nilai USD 50,85 miliar pada tahun 2023 dan diproyeksikan tumbuh dengan CAGR sebesar 5,9% dari 2024 hingga 2030 [1].

Fenomena ini menciptakan apa yang disebut "paradoks pilihan" di mana konsumen justru kesulitan memilih karena terlalu banyaknya pilihan yang tersedia [2]. Selain itu, tantangan lain dalam pemilihan parfum adalah sifat parfum yang sangat subjektif dan sulit dideskripsikan secara tekstual. Aroma yang sama dapat diinterpretasikan berbeda oleh setiap individu berdasarkan pengalaman dan preferensi pribadi mereka.

Sistem rekomendasi parfum hadir sebagai solusi untuk mengatasi masalah ini. Dengan memanfaatkan teknologi machine learning, sistem dapat menganalisis pola preferensi pengguna serta karakteristik parfum untuk memberikan rekomendasi yang personal dan relevan. Penelitian oleh Hussain et al. (2022) menunjukkan bahwa sistem rekomendasi dapat meningkatkan kepuasan pelanggan hingga 27% dan meningkatkan penjualan produk hingga 35% dalam industri e-commerce [3].

**Referensi**:

- [1] Grand View Research, "Perfume Market Size, Share & Trends Analysis Report By Product, By End User (Men, Women), By Distribution Channel (Offline Retail, Online Retail), By Region, And Segment Forecasts, 2024 - 2030," 2023.
- [2] Schwartz, B., "The Paradox of Choice: Why More Is Less," HarperCollins Publishers, 2004.
- [3] Hussain, A., Shahzad, S. K., & Hassan, F., "Impact of Personalized Recommendation Systems on Consumer Behavior in E-commerce," Journal of Marketing Research, vol. 59(2), pp. 231-245, 2022.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah pernyataan masalah yang akan diselesaikan:

- Bagaimana cara mengembangkan sistem rekomendasi yang dapat menyarankan parfum dengan karakteristik serupa berdasarkan preferensi pengguna?
- Bagaimana cara mengidentifikasi dan mengkuantifikasi kesamaan antar parfum berdasarkan deskripsi dan notes aroma mereka?
- Bagaimana cara menghasilkan rekomendasi yang relevan dalam hal aroma dan karakteristik parfum meskipun berasal dari brand yang berbeda?

### Goals

Berikut adalah tujuan yang ingin dicapai untuk menyelesaikan pernyataan masalah di atas:

- Mengembangkan sistem rekomendasi berbasis konten (content-based filtering) yang dapat memberikan rekomendasi parfum berdasarkan karakteristik dan deskripsi parfum.
- Mengimplementasikan teknik pemrosesan bahasa alami (NLP) untuk mengekstrak dan mengkuantifikasi fitur dari deskripsi dan notes aroma parfum.
- Menghasilkan rekomendasi parfum yang beragam namun masih relevan dengan preferensi aroma pengguna, tidak terbatas pada brand tertentu.

### Solution statements

Untuk mencapai tujuan di atas, berikut adalah pendekatan solusi yang akan diimplementasikan:

#### Content-Based Filtering dengan TF-IDF

- Menggunakan TF-IDF Vectorizer untuk mengekstrak fitur penting dari deskripsi dan notes aroma parfum.
- Menghitung similarity score antara parfum menggunakan cosine similarity.
- Merekomendasikan parfum dengan similarity score tertinggi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset parfum yang berisi informasi tentang berbagai parfum beserta detail karakteristiknya. Dataset ini terdiri dari 2.191 entri parfum dengan 5 variabel utama yang memberikan informasi komprehensif tentang setiap parfum. Dataset yang digunakan dalam proyek ini diambil dari Kaggle dan dapat diakses melalui tautan berikut: [Perfume Recommendation Dataset](https://www.kaggle.com/datasets/nandini1999/perfume-recommendation-dataset/).

Berikut adalah variabel-variabel pada dataset parfum:

1. `Name`: Nama parfum (2.184 nilai unik, 0% missing value). Variabel ini mengidentifikasi setiap parfum secara unik.
2. `Brand`: Merek produsen parfum (249 nilai unik, 0% missing value). Menunjukkan keragaman produsen parfum dalam dataset.
3. `Description`: Deskripsi tekstual tentang parfum (2.167 nilai unik, 0% missing value). Berisi narasi tentang parfum, inspirasi, dan karakteristik utamanya. Penting untuk analisis sentimen dan ekstraksi fitur tekstual.
4. `Notes`: Komposisi aroma dalam parfum (2.053 nilai unik, 4% missing value). Menjelaskan bahan-bahan yang membentuk aroma parfum.
5. `Image URL`: URL gambar produk parfum (2.191 nilai unik, 0% missing value). Semua parfum memiliki gambar unik (distinct 100%).

Dari dataset, kita dapat melihat bahwa hampir semua parfum memiliki nama, brand, deskripsi, dan notes yang unik, menunjukkan keberagaman data yang tinggi. Namun, terdapat sekitar 4% data missing pada kolom Notes, yang perlu ditangani pada tahap data preparation.

### Exploratory Data Analysis

#### 1. Distribusi Merek Parfum

Melihat distribusi merek parfum akan membantu memahami keberagaman dan keseimbangan dataset.

```py
# Menampilkan distribusi merek parfum teratas
plt.figure(figsize=(12, 8))
brand_counts = df['Brand'].value_counts().head(20)
sns.barplot(x=brand_counts.values, y=brand_counts.index, palette='viridis')
plt.title('Top 20 Perfume Brands', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Brand', fontsize=12)
plt.tight_layout()
plt.show()

# Menghitung persentase dari total untuk brand terpopuler
top_brands_percentage = (brand_counts.sum() / len(df)) * 100
print(f"Persentase parfum dari 20 brand teratas: {top_brands_percentage:.2f}%")
```

**Insight:**

- Dari analisis, terlihat bahwa distribusi merek parfum sangat tidak merata. Top 20 merek mencakup sekitar 42% dari total dataset.
- Merek seperti PRIN, Jo Malone, dan Tom Ford memiliki representasi yang sangat tinggi dalam dataset.
- Hal ini menunjukkan potensi bias dalam rekomendasi jika tidak ditangani dengan baik, karena sistem mungkin cenderung merekomendasikan merek-merek yang lebih banyak muncul.

#### 2. Wordcloud dari Deskripsi Parfum

Wordcloud membantu memvisualisasikan kata-kata yang paling sering muncul dalam deskripsi parfum.

```py
# Generate wordcloud dari kolom Description
from wordcloud import WordCloud

# Gabungkan semua deskripsi
all_descriptions = ' '.join(df['Description'].dropna())

# Buat wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_descriptions)

# Tampilkan wordcloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Perfume Descriptions', fontsize=15)
plt.tight_layout()
plt.show()
```

**Insight:**

- Kata-kata yang paling sering muncul dalam deskripsi parfum adalah "scent", "fragrance", "notes", "floral", dan "perfume".
- Terdapat banyak kata deskriptif sensori seperti "sweet", "warm", "fresh", dan "woody" yang menggambarkan karakteristik aroma.
- Banyak deskripsi juga menggunakan kata-kata emosional dan simbolik seperti "beautiful", "luxury", "sensual" untuk membangun narasi tentang parfum.
- Ini menunjukkan bahwa fitur tekstual dari deskripsi dapat sangat berguna untuk memahami karakteristik parfum.

#### 3. Wordcloud dari Notes Parfum

```py
# Generate wordcloud dari kolom Notes
all_notes = ' '.join(df['Notes'].dropna())

# Buat wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_notes)

# Tampilkan wordcloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Notes in Perfumes', fontsize=15)
plt.tight_layout()
plt.show()
```

**Insight:**

- Bahan-bahan yang paling sering digunakan dalam parfum termasuk "rose", "vanilla", "musk", "jasmine", dan "bergamot".
- Terdapat keseimbangan antara aroma floral (bunga), fruity (buah), woody (kayu), dan spicy (rempah).
- Beberapa bahan premium seperti "oud", "amber", dan "vetiver" juga sering muncul, menunjukkan kehadiran parfum-parfum kelas atas dalam dataset.
- Notes ini akan menjadi fitur penting dalam sistem rekomendasi berbasis konten.

#### 4. Most Common Perfume Notes

Melihat notes yang paling umum digunakan dalam parfum dapat memberikan wawasan tentang preferensi aroma di pasar.

```py

```

**Insight:**

#### 5. Analisis Panjang Deskripsi dan Jumlah Notes

Menganalisis panjang deskripsi dan jumlah notes dapat memberikan wawasan tentang kelengkapan dan kedetailan informasi di dataset.

```py
# Hitung panjang deskripsi dan jumlah notes
df['Description_Length'] = df['Description'].apply(lambda x: len(str(x).split()) if not pd.isna(x) else 0)
df['Notes_Count'] = df['Notes'].apply(lambda x: len(str(x).split(',')) if not pd.isna(x) else 0)

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panjang deskripsi
sns.histplot(df['Description_Length'], kde=True, ax=axes[0], bins=30, color='skyblue')
axes[0].set_title('Distribution of Description Length (Word Count)')
axes[0].set_xlabel('Number of Words')
axes[0].set_ylabel('Frequency')

# Jumlah notes
sns.histplot(df['Notes_Count'], kde=True, ax=axes[1], bins=30, color='salmon')
axes[1].set_title('Distribution of Notes Count')
axes[1].set_xlabel('Number of Notes')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Statistik deskriptif
description_stats = df['Description_Length'].describe()
notes_stats = df['Notes_Count'].describe()

print("Statistics for Description Length:")
print(description_stats)
print("\nStatistics for Notes Count:")
print(notes_stats)

# Korelasi antara panjang deskripsi dan jumlah notes
correlation = df['Description_Length'].corr(df['Notes_Count'])
print(f"\nKorelasi antara panjang deskripsi dan jumlah notes: {correlation:.3f}")
```

**Insight:**

- Panjang deskripsi parfum bervariasi dengan rata-rata sekitar 60-70 kata. Beberapa parfum memiliki deskripsi yang sangat pendek (<20 kata) sementara yang lain sangat detail (>150 kata).
- Jumlah notes juga bervariasi dengan rata-rata sekitar 5-7 notes per parfum. Beberapa parfum simpel hanya memiliki 1-2 notes sementara yang kompleks bisa memiliki >15 notes.
- Terdapat korelasi positif lemah antara panjang deskripsi dan jumlah notes, menunjukkan bahwa parfum dengan komposisi yang lebih kompleks cenderung memiliki deskripsi yang lebih panjang.
- Parfum-parfum premium dan niche cenderung memiliki deskripsi yang lebih panjang dan notes yang lebih banyak dibandingkan parfum massal.
Insight ini menunjukkan perlunya normalisasi dalam pemrosesan teks untuk menangani variasi panjang deskripsi dan kompleksitas notes.

## Data Preparation

Dalam tahap persiapan data, beberapa teknik preprocessing diterapkan untuk memastikan data dalam kondisi optimal untuk pemodelan machine learning:

### 1. Mengatasi Missing Values

Dataset memiliki sekitar 4% missing values pada kolom Notes yang perlu ditangani sebelum pemodelan.

```py
# Memeriksa missing values
missing_values = df.isna().sum()
print("Missing Values per Kolom:")
print(missing_values)
print(f"\nPersentase missing values pada kolom Notes: {missing_values['Notes']/len(df)*100:.2f}%")

# Menghapus baris dengan missing values jika jumlahnya sedikit
df_clean = df.dropna(subset=['Notes'])
print(f"Jumlah data setelah menghapus missing values: {len(df_clean)}")
```

**Alasan Penanganan:**

- Menghapus baris dengan missing values adalah pendekatan yang paling tepat karena jumlah missing values relatif kecil (4%) dan Notes adalah fitur krusial untuk sistem rekomendasi parfum.
- Mengisi missing values dengan string placeholder atau mengekstraksi dari deskripsi berpotensi menurunkan kualitas rekomendasi karena dapat menyebabkan rekomendasi yang tidak akurat.
- Untuk sistem rekomendasi berbasis konten, kualitas fitur input sangat penting, sehingga lebih baik menghilangkan data tidak lengkap daripada menggunakan perkiraan yang kurang tepat.

### 2. Normalisasi Teks

Normalisasi teks dilakukan untuk menstandarisasi format teks pada kolom Description dan Notes.

```py
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Fungsi untuk membersihkan dan menormalisasi teks
def normalize_text(text):
    if pd.isna(text):
        return text
    
    # Konversi ke lowercase
    text = text.lower()
    
    # Hapus karakter khusus dan angka
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Terapkan normalisasi ke kolom Description dan Notes
df_clean['Description_Normalized'] = df_clean['Description'].apply(normalize_text)
df_clean['Notes_Normalized'] = df_clean['Notes'].apply(normalize_text)

# Contoh hasil normalisasi
print("Sebelum normalisasi:")
print(df_clean['Description'].iloc[0])
print("\nSetelah normalisasi:")
print(df_clean['Description_Normalized'].iloc[0])
```

**Alasan Normalisasi:**

- Normalisasi teks penting untuk mengurangi noise dan variasi yang tidak relevan dalam data tekstual.
- Konversi ke lowercase memastikan konsistensi kasus dan menghindari perbedaan karena kapitalisasi.
- Penghapusan karakter khusus dan angka mengurangi noise yang tidak memberikan informasi semantik tentang parfum.
- Normalisasi teks meningkatkan kualitas ekstraksi fitur dan perhitungan similarity pada tahap berikutnya.

### 3. Tokenisasi dan Stopword Removal

Tokenisasi memecah teks menjadi token (kata-kata individual) dan stopword removal menghilangkan kata-kata umum yang tidak memberikan informasi diskriminatif.

```py
# Download stopwords if not already downloaded
nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

# Tambahkan custom stopwords yang spesifik untuk domain parfum
custom_stopwords = {'perfume', 'fragrance', 'scent', 'smell', 'notes', 'cologne', 'eau', 'de', 'parfum'}
all_stopwords = english_stopwords.union(custom_stopwords)

# Fungsi untuk tokenisasi dan menghapus stopwords
def tokenize_and_clean(text):
    if pd.isna(text):
        return []
    
    # Tokenisasi menggunakan NLTK
    tokens = word_tokenize(text)
    
    # Hapus stopwords
    tokens = [token for token in tokens if token not in all_stopwords]
    
    return tokens

# Terapkan tokenisasi dan stopword removal
df_clean['Description_Tokens'] = df_clean['Description_Normalized'].apply(tokenize_and_clean)
df_clean['Notes_Tokens'] = df_clean['Notes_Normalized'].apply(tokenize_and_clean)

# Gabungkan tokens menjadi teks bersih
df_clean['Description_Clean'] = df_clean['Description_Tokens'].apply(lambda x: ' '.join(x))
df_clean['Notes_Clean'] = df_clean['Notes_Tokens'].apply(lambda x: ' '.join(x))

# Contoh hasil tokenisasi dan pembersihan
print("Hasil tokenisasi dan pembersihan deskripsi parfum pertama:")
print(df_clean['Description_Tokens'].iloc[0][:20])  # Tampilkan 20 token pertama
```

**Alasan Tokenisasi dan Stopword Removal:**

- Tokenisasi adalah langkah fundamental dalam NLP yang memungkinkan analisis pada level kata.
- Stopword removal penting untuk mengurangi dimensi dan meningkatkan efisiensi pemrosesan.
- Kata-kata seperti "the", "and", "in" muncul dengan frekuensi tinggi tetapi tidak memberikan informasi diskriminatif tentang karakteristik parfum.
- Untuk domain parfum, beberapa kata umum seperti "perfume", "fragrance", dan "scent" juga dianggap sebagai stopwords karena tidak membantu membedakan antara parfum.

### 4. Kombinasi Deskripsi dan Notes

Untuk memberikan representasi yang lebih komprehensif tentang parfum, kita akan menggabungkan deskripsi dan notes.

```py
# Gabungkan deskripsi dan notes untuk representasi yang lebih kaya
df_clean['Combined_Features'] = df_clean['Description_Clean'] + ' ' + df_clean['Notes_Clean']

# Terapkan bobot yang lebih tinggi untuk notes (2x) dibandingkan deskripsi
df_clean['Weighted_Features'] = df_clean['Description_Clean'] + ' ' + df_clean['Notes_Clean'] + ' ' + df_clean['Notes_Clean']

# Contoh hasil fitur gabungan
print("Fitur gabungan untuk parfum pertama:")
print(df_clean['Combined_Features'].iloc[0][:200])  # Tampilkan 200 karakter pertama
```

**Alasan Kombinasi:**

- Menggabungkan deskripsi dan notes memungkinkan model untuk mempertimbangkan kedua aspek parfum dalam rekomendasi.
- Pemberian bobot lebih tinggi pada notes (diulang dua kali) karena notes lebih langsung mencerminkan karakteristik aroma parfum dibandingkan deskripsi yang lebih naratif.
- Fitur gabungan yang diperkaya meningkatkan kemampuan sistem untuk mengenali kesamaan antar parfum berdasarkan baik karakteristik sensorik maupun konteks penggunaannya.

## Modeling

Pada tahap ini, kita akan mengimplementasikan model sistem rekomendasi berbasis konten (Content-Based Filtering) dengan menggunakan teknik TF-IDF dan Cosine Similarity untuk merekomendasikan parfum dengan karakteristik serupa.

### Model Content-Based Filtering dengan TF-IDF

```py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    min_df=2,              # Abaikan kata yang muncul di kurang dari 2 dokumen
    max_df=0.95,           # Abaikan kata yang muncul di lebih dari 95% dokumen
    max_features=5000,     # Batasi jumlah fitur (kata) maksimum
    ngram_range=(1, 2)     # Gunakan unigram dan bigram
)

# Fit dan transform pada fitur gabungan terbobot
tfidf_matrix = tfidf.fit_transform(df_clean['Weighted_Features'])

# Hitung cosine similarity antara semua parfum
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Indeks untuk memetakan kembali ke nama parfum
indices = pd.Series(df_clean.index, index=df_clean['Name']).drop_duplicates()

# Fungsi untuk mendapatkan rekomendasi berdasarkan nama parfum
def get_recommendations(name, cosine_sim=cosine_sim, df=df_clean, indices=indices, top_n=10):
    # Dapatkan indeks parfum yang dicari
    idx = indices[name]

    # Dapatkan skor kesamaan untuk semua parfum dengan parfum yang dicari
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Urutkan parfum berdasarkan skor kesamaan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Dapatkan top N parfum yang paling mirip (tidak termasuk diri sendiri di indeks 0)
    sim_scores = sim_scores[1:top_n+1]

    # Dapatkan indeks parfum
    perfume_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # Buat dataframe hasil rekomendasi
    recommendations = pd.DataFrame({
        'Name': df['Name'].iloc[perfume_indices].values,
        'Brand': df['Brand'].iloc[perfume_indices].values,
        'Notes': df['Notes'].iloc[perfume_indices].values,
        'Similarity Score': similarity_scores
    })

    return recommendations

# Contoh rekomendasi untuk parfum tertentu
sample_perfume = df_clean['Name'].iloc[42]  # Ambil parfum sampel
print(f"Rekomendasi untuk parfum: {sample_perfume} (Brand: {df_clean['Brand'].iloc[42]})")
print(f"Notes: {df_clean['Notes'].iloc[42]}")
print("\nRekomendasi Parfum Serupa:")
recommendations = get_recommendations(sample_perfume)
print(recommendations)
```

### Penjelasan Model

#### 1. TF-IDF Vectorization

- TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah teks menjadi vektor numerik.
- Parameter min_df=2 menghilangkan kata yang sangat jarang (muncul di kurang dari 2 dokumen).
- Parameter max_df=0.95 menghilangkan kata yang terlalu umum (muncul di lebih dari 95% dokumen).
- Parameter ngram_range=(1, 2) memungkinkan penangkapan unigram (kata tunggal) dan bigram (2 kata berurutan).

#### 2. Cosine Similarity

- Cosine similarity mengukur kesamaan antara dua vektor dengan menghitung kosinus sudut di antara keduanya.
- Nilai berkisar dari 0 (tidak mirip sama sekali) hingga 1 (identik).
- Formula: $\text{cosine similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}||\mathbf{B}|}$

#### 3. Sistem Rekomendasi

- Untuk setiap parfum yang menjadi input, sistem menghitung kesamaan dengan semua parfum lain dalam dataset.
- Sistem kemudian mengurutkan parfum berdasarkan skor kesamaan dan mengambil N parfum teratas.
- Hasil rekomendasi menampilkan nama parfum, brand, notes, dan skor kesamaan.

### Parameter Penting

#### Penggunaan Fitur Gabungan Terbobot

- Fitur Weighted_Features memberikan bobot lebih tinggi pada notes karena notes lebih langsung mencerminkan karakteristik aroma parfum.
- Hal ini meningkatkan kualitas rekomendasi untuk kasus di mana pengguna lebih mementingkan kesamaan aroma daripada narasi pemasaran.

#### Pemilihan n-gram

Penggunaan bigram memungkinkan penangkapan frasa penting seperti "amber wood", "rose water", atau "citrus bergamot" yang memberikan informasi lebih dari sekadar kata individual.

#### Min_df dan Max_df

Parameter ini membantu mengurangi noise dari kata-kata yang terlalu jarang atau terlalu umum, meningkatkan kualitas fitur.

## Evaluation

Untuk mengevaluasi kinerja model rekomendasi berbasis konten yang telah dikembangkan, kita akan menggunakan beberapa metrik evaluasi yang sesuai untuk sistem rekomendasi:

### Cosine Similarity Score

Cosine similarity adalah metrik utama yang digunakan dalam model ini untuk mengukur kesamaan antara parfum. Nilai berkisar dari 0 (tidak mirip sama sekali) hingga 1 (identik).

```py
# Evaluasi distribusi cosine similarity scores
def analyze_similarity_distribution(cosine_sim):
    # Ambil nilai similarity dari matriks (tidak termasuk diagonal yang selalu 1)
    similarity_values = []
    n = cosine_sim.shape[0]
    for i in range(n):
        for j in range(i+1, n):  # Hanya ambil nilai di atas diagonal
            similarity_values.append(cosine_sim[i, j])
    
    # Analisis statistik
    avg_similarity = np.mean(similarity_values)
    median_similarity = np.median(similarity_values)
    std_similarity = np.std(similarity_values)
    
    print(f"Statistik Cosine Similarity:")
    print(f"- Rata-rata: {avg_similarity:.4f}")
    print(f"- Median: {median_similarity:.4f}")
    print(f"- Standar Deviasi: {std_similarity:.4f}")
    print(f"- Min: {min(similarity_values):.4f}")
    print(f"- Max: {max(similarity_values):.4f}")
    
    # Visualisasi distribusi
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_values, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(avg_similarity, color='red', linestyle='--', label=f'Mean: {avg_similarity:.4f}')
    plt.title('Distribution of Cosine Similarity Scores')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Analisis distribusi similarity
analyze_similarity_distribution(cosine_sim)
```