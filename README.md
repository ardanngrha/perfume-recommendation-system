# Laporan Proyek Machine Learning - Ardana Aldhizuma Nugraha

## Project Overview

Parfum telah menjadi bagian penting dalam kehidupan manusia sejak ribuan tahun lalu, bukan hanya sebagai penyegar bau, tetapi juga sebagai bentuk ekspresi identitas pribadi. Di era modern, industri parfum telah berkembang pesat dengan ribuan produk yang tersedia di pasaran, menyulitkan konsumen untuk menemukan aroma yang sesuai dengan preferensi mereka. Menurut penelitian yang dilakukan oleh Grand View Research, pasar global parfum mencapai nilai USD 50,85 miliar pada tahun 2023 dan diproyeksikan tumbuh dengan CAGR sebesar 5,9% dari 2024 hingga 2030 [1].

Fenomena ini menciptakan apa yang disebut "paradoks pilihan" di mana konsumen justru kesulitan memilih karena terlalu banyaknya pilihan yang tersedia [2]. Selain itu, tantangan lain dalam pemilihan parfum adalah sifat parfum yang sangat subjektif dan sulit dideskripsikan secara tekstual. Aroma yang sama dapat diinterpretasikan berbeda oleh setiap individu berdasarkan pengalaman dan preferensi pribadi mereka.

Sistem rekomendasi parfum hadir sebagai solusi untuk mengatasi masalah ini. Dengan memanfaatkan teknologi machine learning, sistem dapat menganalisis pola karakteristik parfum untuk memberikan rekomendasi yang personal dan relevan. Penelitian oleh Hussain et al. (2022) menunjukkan bahwa sistem rekomendasi dapat meningkatkan kepuasan pelanggan hingga 27% dan meningkatkan penjualan produk hingga 35% dalam industri e-commerce [3].

**Referensi**:

- [1] Grand View Research, "Perfume Market Size, Share & Trends Analysis Report By Product, By End User (Men, Women), By Distribution Channel (Offline Retail, Online Retail), By Region, And Segment Forecasts, 2024 - 2030," 2023.
- [2] Schwartz, B., "The Paradox of Choice: Why More Is Less," HarperCollins Publishers, 2004.
- [3] Hussain, A., Shahzad, S. K., & Hassan, F., "Impact of Personalized Recommendation Systems on Consumer Behavior in E-commerce," Journal of Marketing Research, vol. 59(2), pp. 231-245, 2022.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah pernyataan masalah yang akan diselesaikan:

- Bagaimana cara mengembangkan sistem rekomendasi yang dapat menyarankan parfum dengan karakteristik aroma serupa berdasarkan preferensi pengguna?
- Bagaimana cara mengidentifikasi dan mengkuantifikasi kesamaan antar parfum berdasarkan karakteristik aroma mereka?
- Bagaimana cara menghasilkan rekomendasi yang relevan dalam hal karakteristik aroma parfum?

### Goals

Berikut adalah tujuan yang ingin dicapai untuk menyelesaikan pernyataan masalah di atas:

- Mengembangkan sistem rekomendasi berbasis konten (content-based filtering) yang dapat memberikan rekomendasi parfum berdasarkan karakteristik aroma parfum.
- Mengimplementasikan teknik pemrosesan bahasa alami (NLP) untuk mengekstrak dan mengkuantifikasi fitur karakteristik aroma parfum.
- Menghasilkan rekomendasi parfum yang beragam namun masih relevan dengan preferensi karakteristik aroma pengguna.

### Solution statements

Untuk mencapai tujuan di atas, berikut adalah pendekatan solusi yang akan diimplementasikan:

- Menggunakan TF-IDF Vectorizer untuk mengekstrak fitur penting dari karakteristik aroma parfum.
- Menghitung similarity score antara parfum menggunakan cosine similarity.
- Merekomendasikan parfum dengan similarity score tertinggi.

Performa sistem rekomendasi akan dievaluasi berdasarkan metrik kesamaan (similarity score) dan keragaman (diversity) dari rekomendasi yang dihasilkan.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset parfum yang berisi informasi tentang berbagai parfum beserta detail karakteristiknya. Dataset ini terdiri dari 2.191 entri parfum dengan 5 variabel utama yang memberikan informasi komprehensif tentang setiap parfum. Dataset yang digunakan dalam proyek ini diambil dari Kaggle dan dapat diakses melalui tautan berikut: [Perfume Recommendation Dataset](https://www.kaggle.com/datasets/nandini1999/perfume-recommendation-dataset/).

Berikut adalah variabel-variabel pada dataset parfum:

**GAMBAR**

1. `Name`: Nama parfum (2.184 nilai unik, 0% missing value). Variabel ini mengidentifikasi setiap parfum secara unik.
2. `Brand`: Merek produsen parfum (249 nilai unik, 0% missing value). Menunjukkan keragaman produsen parfum dalam dataset.
3. `Description`: Deskripsi tekstual tentang parfum (2.167 nilai unik, 0% missing value). Berisi narasi tentang parfum, inspirasi, dan karakteristik utamanya. Penting untuk analisis sentimen dan ekstraksi fitur tekstual.
4. `Notes`: Komposisi karakteristik aroma dalam parfum (2.053 nilai unik, 4% missing value). Menjelaskan bahan-bahan yang membentuk aroma parfum.
5. `Image URL`: URL gambar produk parfum (2.191 nilai unik, 0% missing value). Semua parfum memiliki gambar unik (distinct 100%).

Dari dataset, kita dapat melihat bahwa hampir semua parfum memiliki nama, brand, deskripsi, dan notes yang unik, menunjukkan keberagaman data yang tinggi. Namun, terdapat sekitar 4% data missing pada kolom Notes, yang perlu ditangani pada tahap data preparation.

### Exploratory Data Analysis

#### 1. Distribusi Merek Parfum

Melihat distribusi merek parfum akan membantu memahami keberagaman dan keseimbangan dataset.

```py
plt.figure(figsize=(10, 6))
brand_counts = df['Brand'].value_counts().head(20)
sns.barplot(x=brand_counts.values, y=brand_counts.index, hue=brand_counts.index, palette='viridis', legend=False)
plt.title('Merek Parfum 20 Teratas', fontsize=15)
plt.xlabel('Jumlah', fontsize=12)
plt.ylabel('Merek', fontsize=12)
plt.tight_layout()
plt.show()
```

**GAMBAR**

- Berdasarkan visualisasi, TOM FORD Private Blend menduduki posisi teratas dengan hampir 40 parfum dalam dataset, diikuti oleh Profumum dan Serge Lutens yang memiliki sekitar 35-38 parfum.
- Merek-merek premium seperti BYREDO, Xerjoff, L'Artisan Parfumeur, dan Montale memiliki representasi tinggi, menunjukkan bahwa dataset lebih condong pada segmen parfum kelas atas.

#### 2. Wordcloud dari Aroma Parfum

```py
all_notes = ', '.join(df['Notes'].dropna().astype(str))

all_notes = re.sub(r'[^\w\s,]', '', all_notes.lower())
all_notes_list = [note.strip() for note in all_notes.split(',')]

wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_notes)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()
```

**GAMBAR**

- Wordcloud menunjukkan dominasi beberapa aroma kunci dalam komposisi parfum: "musk", "rose", "vanilla", "patchouli", dan "sandalwood" terlihat sebagai aroma yang paling menonjol dalam dataset.
- Terdapat keseimbangan antara berbagai kategori aroma: floral (rose, jasmine), woody (sandalwood, cedar, oud), gourmand (vanilla), dan earthy (patchouli, vetiver), menunjukkan keragaman komposisi parfum dalam dataset.
- Aroma oriental seperti "amber" dan "incense" memiliki representasi yang signifikan, mencerminkan tren parfumeri kontemporer yang menggabungkan elemen-elemen Timur dalam komposisi.
- Bahan-bahan premium dan langka seperti "oud", "labdanum", dan "benzoin" juga terlihat jelas, menguatkan indikasi bahwa dataset lebih fokus pada parfum kelas atas dan niche.
- Kehadiran bahan-bahan citrus seperti "bergamot" dan "lemon" yang cukup dominan menunjukkan pentingnya aroma segar dalam formulasi parfum modern.

#### 3. Aroma Paling Umum

Melihat notes yang paling umum digunakan dalam parfum dapat memberikan wawasan tentang preferensi aroma di pasar.

```py
notes_counter = Counter(all_notes_list)

all_notes_freq = notes_counter.most_common(15)
all_notes_df = pd.DataFrame(all_notes_freq, columns=['Note', 'Frequency'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Note', hue='Note', data=all_notes_df, palette='viridis', legend=False)
plt.title('Aroma Parfum Paling Umum', fontsize=15)
plt.xlabel('Frekuensi', fontsize=12)
plt.ylabel('Aroma', fontsize=12)
plt.tight_layout()
plt.show()
```

**GAMBAR**

- "musk" merupakan aroma paling dominan dalam dataset dengan frekuensi kemunculan tertinggi (>500 kali), jauh melebihi aroma lainnya, menunjukkan peran pentingnya sebagai bahan pengikat (fixative) dan pemberi karakter dalam formulasi parfum modern.
- Lima aroma teratas didominasi oleh kombinasi dari woody dan oriental notes: "musk", "patchouli", "vanilla", "sandalwood", dan "bergamot", yang menggambarkan preferensi pasar terhadap aroma yang kompleks, hangat, dan tahan lama.
- "Vanilla" dan "patchouli" yang menempati posisi 2-3 teratas menunjukkan tren kuat parfum dengan karakter gourmand (manis) dan earthy (beraroma tanah) dalam industri parfum kontemporer.
- Aroma floral klasik seperti "rose" dan "jasmine" berada di posisi 7-8, membuktikan bahwa meskipun ada inovasi dalam formulasi parfum, bahan-bahan tradisional tetap menjadi fondasi penting.
- Adanya "vetiver", "cedar", dan "leather" dalam 10 besar mencerminkan popularitas parfum maskulin atau unisex dalam dataset, dimana aroma woody dan leathery sangat dihargai.
Distribusi frekuensi yang menurun secara bertahap dari "musk" ke "saffron" menunjukkan konsentrasi terhadap beberapa aroma kunci dalam mayoritas komposisi parfum.

#### 4. Analisis Jumlah Notes

Menganalisis jumlah notes dapat memberikan wawasan tentang kelengkapan dan kedetailan informasi di dataset.

```py
# Hitung jumlah notes
df['Notes_Count'] = df['Notes'].apply(lambda x: len(str(x).split(',')) if not pd.isna(x) else 0)

plt.figure(figsize=(10, 6))
sns.histplot(df['Notes_Count'], bins=30, color='salmon')
plt.title('Distribusi Jumlah Aroma dalam Parfum', fontsize=15)
plt.xlabel('Jumlah Aroma', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

**GAMBAR**

- Histogram pada Image 1 menunjukkan bahwa mayoritas parfum dalam dataset memiliki antara 5-10 aroma dalam komposisinya, dengan puncak distribusi berada di sekitar 6-7 aroma per parfum.
- Terdapat distribusi yang menarik dimana sekitar 100 parfum memiliki jumlah aroma yang sangat rendah (0-1), yang mungkin mengindikasikan parfum minimalis atau data yang kurang lengkap untuk beberapa entri.
- Dua puncak terlihat pada distribusi di sekitar 5-6 aroma dan 9-10 aroma, menunjukkan kemungkinan adanya dua pendekatan umum dalam formulasi parfum: komposisi sederhana dengan fokus pada beberapa bahan utama, dan komposisi kompleks dengan lapisan aroma yang lebih banyak.
- Jumlah parfum dengan lebih dari 15 aroma sangat sedikit, mengindikasikan bahwa parfum dengan komposisi sangat kompleks merupakan minoritas dalam industri.
- Distribusi ini mencerminkan filosofi parfumeri modern yang seimbang antara kompleksitas dan kejelasan karakter aroma, di mana jumlah aroma yang terlalu banyak dapat membuat parfum kehilangan identitasnya.
- Adanya parfum dengan jumlah aroma hingga 30+ (meskipun jarang) menunjukkan keberadaan parfum yang sangat kompleks dan berlapis dalam dataset, biasanya merupakan kreasi dari rumah parfum eksklusif atau parfumer eksperimental.

## Data Preparation

Dalam tahap persiapan data, beberapa teknik preprocessing diterapkan untuk memastikan data dalam kondisi optimal untuk pemodelan machine learning:

### 1. Mengatasi Missing Values

Dataset memiliki sekitar 4% missing values pada kolom Notes yang perlu ditangani sebelum pemodelan.

```py
missing_values = df.isna().sum()
df_clean = df.dropna(subset=['Notes']).copy()
```

**GAMBAR**

- Menghapus baris dengan missing values adalah pendekatan yang paling tepat karena jumlah missing values relatif kecil (4%) dan Notes adalah fitur krusial untuk sistem rekomendasi parfum.

### 2. Normalisasi Teks

Normalisasi teks dilakukan untuk menstandarisasi format teks pada kolom Description dan Notes.

```py
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

# Terapkan normalisasi ke kolom Description dan Notes menggunakan .loc
df_clean.loc[:, 'Notes_Normalized'] = df_clean['Notes'].apply(normalize_text)
```

**GAMBAR**

- Normalisasi teks penting untuk mengurangi noise dan variasi yang tidak relevan dalam data tekstual.
- Konversi ke lowercase memastikan konsistensi kasus dan menghindari perbedaan karena kapitalisasi.
- Penghapusan karakter khusus dan angka mengurangi noise yang tidak memberikan informasi semantik tentang parfum.
- Normalisasi teks meningkatkan kualitas ekstraksi fitur dan perhitungan similarity pada tahap berikutnya.

### 3. Tokenisasi

Tokenisasi memecah teks menjadi token (kata-kata individual).

```py
def tokenize(text):
    if pd.isna(text):
        return []
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    return tokens

# Terapkan tokenisasi dan stopword removal
df_clean.loc[:, 'Notes_Tokens'] = df_clean['Notes_Normalized'].apply(tokenize)

# Gabungkan tokens menjadi teks bersih
df_clean.loc[:, 'Notes_Clean'] = df_clean['Notes_Tokens'].apply(lambda x: ' '.join(x))
```

**GAMBAR**

Tokenisasi adalah langkah fundamental dalam NLP yang memungkinkan analisis pada level kata.

Data siap digunakan untuk pemodelan sistem rekomendasi berbasis konten. Dataset telah dibersihkan dari missing values, dinormalisasi, dan ditokenisasi untuk memastikan kualitas data yang optimal. **Tidak digunakan** stopword removal pada tahap ini karena kita ingin mempertahankan semua kata untuk analisis similarity yang lebih baik.

## Modeling

Pada tahap ini, kita akan mengimplementasikan model sistem rekomendasi berbasis konten (Content-Based Filtering) dengan menggunakan teknik TF-IDF dan Cosine Similarity untuk merekomendasikan parfum dengan karakteristik serupa.

### TF-IDF Vectorization

```py
tfidf_vectorizer = TfidfVectorizer(
  min_df=2,
  max_df=0.9,
  max_features=1000,
  ngram_range=(1, 2)
  )
tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['Notes_Clean'])

print(tfidf_matrix.shape)
```

**GAMBAR**

### Cosine Similarity

```py
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Shape dari Cosine Similarity Matrix: {cosine_sim.shape}")
```

**GAMBAR**

### Sistem Rekomendasi

```py
def recommend_perfume(name, top_n=5):
    idx = df_clean[df_clean['Name'].str.lower() == name.lower()].index
    if len(idx) == 0:
        return f"Parfum '{name}' tidak ditemukan."
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    perfume_indices = [i[0] for i in sim_scores]
    return df_clean[['Name', 'Brand', 'Notes']].iloc[perfume_indices], [i[1] for i in sim_scores]
```

### Top 10 rekomendasi parfum

```py
name = "Tihota Eau de Parfum"
brand = df[df['Name'].str.lower() == name.lower()]['Brand'].values[0]
notes = df[df['Name'].str.lower() == name.lower()]['Notes'].values[0]
recommendations, scores = recommend_perfume(name, top_n=10)

recommendations['Similarity Score'] = scores
print(f"\nRekomendasi parfum mirip dengan: {name} ({brand}) - {notes}")
recommendations
```

**GAMBAR**

### Penjelasan Model

#### 1. TF-IDF Vectorization

- TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah teks menjadi vektor numerik.
- Parameter `min_df=2` menghilangkan kata yang sangat jarang (muncul di kurang dari 2 dokumen). Parameter `max_df=0.9` menghilangkan kata yang terlalu umum (muncul di lebih dari 90% dokumen). Parameter ini membantu mengurangi noise dari kata-kata yang terlalu jarang atau terlalu umum, meningkatkan kualitas fitur.
- Parameter max_features=1000 membatasi jumlah fitur yang diambil menjadi 1000 kata paling penting.
- Parameter ngram_range=(1, 2) memungkinkan penangkapan unigram (kata tunggal) dan bigram (2 kata berurutan). Penggunaan bigram memungkinkan penangkapan frasa penting seperti "amber wood", "rose water", atau "citrus bergamot" yang memberikan informasi lebih dari sekadar kata individual.
  
Kelebihan TF-IDF adalah kemampuannya untuk menangkap pentingnya kata dalam konteks dokumen, sehingga membantu dalam membedakan parfum berdasarkan karakteristik aroma mereka. Dengan menggunakan TF-IDF, kita dapat mengubah deskripsi dan notes parfum menjadi representasi numerik yang dapat digunakan untuk menghitung kesamaan antar parfum.

Kekurangan TF-IDF adalah bahwa ia tidak mempertimbangkan urutan kata, sehingga informasi tentang konteks dan struktur kalimat hilang. Namun, dalam konteks sistem rekomendasi parfum, fokus utama adalah pada kesamaan karakteristik aroma, sehingga TF-IDF tetap menjadi pilihan yang baik.

#### 2. Cosine Similarity

- Cosine similarity mengukur kesamaan antara dua vektor dengan menghitung kosinus sudut di antara keduanya.
- Nilai berkisar dari 0 (tidak mirip sama sekali) hingga 1 (identik).

#### 3. Sistem Rekomendasi

- Untuk setiap parfum yang menjadi input, sistem menghitung kesamaan dengan semua parfum lain dalam dataset.
- Sistem kemudian mengurutkan parfum berdasarkan skor kesamaan dan mengambil N parfum teratas.
- Hasil rekomendasi menampilkan nama parfum, brand, notes, dan skor kesamaan.
- Pada contoh di atas, sistem merekomendasikan 10 parfum teratas yang paling mirip dengan parfum "Tihota Eau de Parfum".
- Hasil rekomendasi menunjukkan bahwa parfum-parfum yang direkomendasikan memiliki kesamaan yang tinggi dengan parfum input, berdasarkan karakteristik aroma mereka.

## Evaluation

Untuk mengevaluasi kinerja model rekomendasi berbasis konten yang telah dikembangkan, kita akan menggunakan beberapa metrik evaluasi yang sesuai untuk sistem rekomendasi:

### 1. Similarity Score

Similarity adalah metrik utama yang digunakan dalam model ini untuk mengukur kesamaan antara parfum. Nilai berkisar dari 0 (tidak mirip sama sekali) hingga 1 (identik).

Formula untuk menghitung similarity score adalah sebagai berikut:

$\text{Cosine Similarity} = \frac{A \cdot B}{||A|| \cdot ||B||}$

di mana:

- $A$ dan $B$ adalah vektor representasi dari dua parfum yang dibandingkan.
- $||A||$ dan $||B||$ adalah norma (magnitudo) dari vektor $A$ dan $B$.
- $A \cdot B$ adalah hasil kali dot antara dua vektor.

### 2. Diversity

Diversity mengukur seberapa beragam rekomendasi yang diberikan oleh sistem. Dalam konteks ini, kita dapat menggunakan metrik seperti Jaccard Similarity untuk mengukur kesamaan antara notes parfum dalam rekomendasi.

Formula untuk menghitung Jaccard Similarity adalah sebagai berikut:

$\text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|}$

di mana:

- $|A \cap B|$ adalah jumlah elemen yang ada di kedua set (kesamaan).
- $|A \cup B|$ adalah jumlah elemen yang ada di set $A$ dan $B$ (total elemen).
- $|A|$ dan $|B|$ adalah jumlah elemen dalam set $A$ dan $B$ masing-masing.

### Rangkuman Evaluasi

**GAMBAR**

## Conclusion

Berdasarkan hasil pengembangan dan evaluasi model rekomendasi parfum berbasis konten, beberapa kesimpulan penting dapat ditarik:

### Keterkaitan dengan Business Understanding

- Solusi untuk "Paradoks Pilihan": Sistem rekomendasi parfum yang dikembangkan berhasil mengatasi permasalahan "paradoks pilihan" dengan menyederhanakan proses pencarian parfum. Dari ribuan pilihan parfum yang tersedia di pasar, sistem mampu menyarankan 5-10 parfum yang paling relevan dengan preferensi karakteristik aroma pengguna.
- Kuantifikasi Kesamaan Parfum: Dengan mengimplementasikan TF-IDF dan cosine similarity, model berhasil mengkuantifikasi kesamaan antar parfum berdasarkan karakteristik aroma mereka. Nilai similarity score berkisar antara 0.2161 hingga 0.5213 (untuk kasus Tihota Eau de Parfum), menunjukkan kemampuan model dalam membedakan tingkat kesamaan antar parfum.
- Rekomendasi Relevan dengan Preferensi Pengguna: Sebagaimana terlihat dari hasil rekomendasi untuk parfum "Tihota Eau de Parfum" yang memiliki notes "Vanilla bean, musks", sistem berhasil merekomendasikan parfum-parfum dengan karakteristik aroma serupa seperti "Fat Electrician Eau de Parfum" yang mengandung "vetiver, vanilla bean, opoponax and myrrh".

### Dampak Solusi

- Peningkatan Pengalaman Pengguna: Sistem rekomendasi parfum dapat meningkatkan pengalaman belanja konsumen dengan menyediakan rekomendasi yang personal dan relevan, mengurangi waktu pencarian, dan mengurangi "pilihan yang berlebihan".
- Potensi Peningkatan Penjualan: Berdasarkan literatur yang dikaji, implementasi sistem rekomendasi yang efektif dapat meningkatkan penjualan hingga 35% dan kepuasan pelanggan hingga 27%. Sistem yang dikembangkan berpotensi memberikan dampak positif serupa pada bisnis parfum.
- Diversifikasi Brand dan Notes: Hasil evaluasi menunjukkan tingkat diversity yang baik dengan 7 brand unik dari 10 rekomendasi (brand diversity ratio 0.70) dan 55 notes unik dengan rata-rata 7.40 notes per rekomendasi. Ini menunjukkan keberhasilan sistem dalam memberikan rekomendasi yang beragam namun tetap relevan.
- Kualitas Rekomendasi: Berdasarkan evaluasi similarity score, sistem mampu memberikan rekomendasi dengan skor kesamaan yang tinggi (rata-rata 0.36) dan variasi yang cukup baik (variance 0.02). Ini menunjukkan bahwa sistem dapat memberikan rekomendasi yang relevan dan bervariasi.
  
### Rekomendasi untuk Pengembangan Selanjutnya

- Penggunaan Model Hybrid: Menggabungkan sistem rekomendasi berbasis konten dengan sistem rekomendasi berbasis kolaboratif (collaborative filtering) dapat meningkatkan akurasi dan relevansi rekomendasi.
- Penerapan Deep Learning: Menggunakan model deep learning seperti neural networks untuk menangkap pola yang lebih kompleks dalam data dapat meningkatkan performa sistem rekomendasi.
- Peningkatan Data: Mengumpulkan lebih banyak data parfum, termasuk ulasan pengguna dan rating, dapat meningkatkan kualitas model dan memberikan rekomendasi yang lebih personal.
