#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity


# # Data Gathering

# In[2]:


df = pd.read_csv('final_perfume_data.csv', encoding = 'unicode_escape')
df.head()


# In[3]:


(df.info())


# Dataset yang digunakan berisi 2.191 entri parfum dengan 5 variabel utama:
# 1. `Name`: Nama parfum (2.184 nilai unik)
# 2. `Brand`: Merek produsen parfum (249 nilai unik)
# 3. `Description`: Deskripsi tekstual tentang parfum (2.167 nilai unik)
# 4. `Notes`: Komposisi karakteristik aroma dalam parfum (2.053 nilai unik, 4% missing value)
# 5. `Image URL`: URL gambar produk parfum (2.191 nilai unik)

# # Data Understanding

# ### 1. Distribusi Merek

# In[4]:


# Menampilkan distribusi merek parfum teratas
plt.figure(figsize=(10, 6))
brand_counts = df['Brand'].value_counts().head(20)
sns.barplot(x=brand_counts.values, y=brand_counts.index, hue=brand_counts.index, palette='viridis', legend=False)
plt.title('Merek Parfum 20 Teratas', fontsize=15)
plt.xlabel('Jumlah', fontsize=12)
plt.ylabel('Merek', fontsize=12)
plt.tight_layout()
plt.show()

# Menghitung persentase dari total untuk brand terpopuler
top_brands_percentage = (brand_counts.sum() / len(df)) * 100
print(f"Persentase parfum dari 20 merek teratas: {top_brands_percentage:.2f}%")


# In[5]:


print("Brand parfum terpopuler:")
print(brand_counts.head(10))


# Berdasarkan visualisasi, TOM FORD Private Blend menduduki posisi teratas dengan hampir 40 parfum dalam dataset, diikuti oleh Profumum dan Serge Lutens. Merek-merek premium seperti BYREDO, Xerjoff, L'Artisan Parfumeur, dan Montale memiliki representasi tinggi, menunjukkan bahwa dataset lebih condong pada segmen parfum kelas atas.

# ### 2. Word Cloud Aroma

# In[6]:


# Get all notes as a single string
all_notes = ', '.join(df['Notes'].dropna().astype(str))

# Clean and tokenize
all_notes = re.sub(r'[^\w\s,]', '', all_notes.lower())
all_notes_list = [note.strip() for note in all_notes.split(',')]

# Buat wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_notes)

# Tampilkan wordcloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()


# Wordcloud menunjukkan dominasi beberapa aroma kunci dalam komposisi parfum: "musk", "rose", "vanilla", "patchouli", dan "sandalwood" terlihat sebagai aroma yang paling menonjol dalam dataset. Terdapat keseimbangan antara berbagai kategori aroma: floral (rose, jasmine), woody (sandalwood, cedar, oud), gourmand (vanilla), dan earthy (patchouli, vetiver), menunjukkan keragaman komposisi parfum dalam dataset.

# ### 3. Aroma Terpopuler

# In[7]:


# Plot the most common notes overall
notes_counter = Counter(all_notes_list)

all_notes_freq = notes_counter.most_common(15)
all_notes_df = pd.DataFrame(all_notes_freq, columns=['Note', 'Frequency'])


# In[8]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Note', hue='Note', data=all_notes_df, palette='viridis', legend=False)
plt.title('Aroma Parfum Paling Umum', fontsize=15)
plt.xlabel('Frekuensi', fontsize=12)
plt.ylabel('Aroma', fontsize=12)
plt.tight_layout()
plt.show()


# In[9]:


print("Aroma yang paling umum:")
for note, freq in all_notes_freq:
    print(f"{note}: {freq}")


# "musk" merupakan aroma paling dominan dalam dataset dengan frekuensi kemunculan tertinggi (>500 kali). Lima aroma teratas didominasi oleh kombinasi dari woody dan oriental notes: "musk", "patchouli", "vanilla", "sandalwood", dan "bergamot", yang menggambarkan preferensi pasar terhadap aroma yang kompleks, hangat, dan tahan lama.

# ### 4. Distribusi Jumlah Aroma

# In[10]:


# Hitung jumlah notes
df['Notes_Count'] = df['Notes'].apply(lambda x: len(str(x).split(',')) if not pd.isna(x) else 0)

# Visualisasi distribusi jumlah notes
plt.figure(figsize=(10, 6))
sns.histplot(df['Notes_Count'], bins=30, color='salmon')
plt.title('Distribusi Jumlah Aroma dalam Parfum', fontsize=15)
plt.xlabel('Jumlah Aroma', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Statistik deskriptif untuk jumlah aroma
notes_stats = df['Notes_Count'].describe()
print("Statistik Deskriptif untuk Jumlah Aroma:")
print(notes_stats)

# Tambahan analisis untuk aroma count
print(f"\nParfum dengan nol aroma: {(df['Notes_Count'] == 0).sum()} ({(df['Notes_Count'] == 0).sum()/len(df)*100:.2f}%)")
print(f"Parfum dengan lebih dari 15 aroma: {(df['Notes_Count'] > 15).sum()} ({(df['Notes_Count'] > 15).sum()/len(df)*100:.2f}%)")
print(f"Median jumlah aroma per parfum: {df['Notes_Count'].median()}")


# Mayoritas parfum dalam dataset memiliki antara 5-10 aroma dalam komposisinya, dengan puncak distribusi berada di sekitar 6-7 aroma per parfum. Distribusi ini mencerminkan filosofi parfumeri modern yang seimbang antara kompleksitas dan kejelasan karakter aroma.

# # Data Preparation

# ### 1. Penanganan Missing Value

# In[11]:


missing_values = df.isna().sum()
print("Missing Values per Kolom:")
print(missing_values)
print(f"\nPersentase missing values pada kolom Notes: {missing_values['Notes']/len(df)*100:.2f}%")


# In[12]:


df_clean = df.dropna(subset=['Notes']).copy()


# In[13]:


df_clean.info()


# Menghapus baris dengan missing values adalah pendekatan yang paling tepat karena jumlah missing values relatif kecil (4%) dan aroma adalah fitur krusial untuk sistem rekomendasi parfum.

# ### 2. Normalisasi Teks

# In[14]:


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


# In[15]:


# Terapkan normalisasi ke kolom Description dan Notes menggunakan .loc
df_clean.loc[:, 'Notes_Normalized'] = df_clean['Notes'].apply(normalize_text)

# Contoh hasil normalisasi
print("Sebelum normalisasi:")
df_clean['Notes'].iloc[1]


# In[16]:


print("\nSetelah normalisasi:")
df_clean['Notes_Normalized'].iloc[1]


# In[17]:


df_clean.head()


# In[18]:


df_clean.info()


# Data siap digunakan untuk pemodelan sistem rekomendasi berbasis konten. Dataset telah dibersihkan dari missing values dan dinormalisasi untuk memastikan kualitas data yang optimal.

# # Modeling

# ### TF-IDF Vectorization

# In[19]:


tfidf_vectorizer = TfidfVectorizer(
  min_df=2,
  max_df=0.9,
  max_features=1000,
  ngram_range=(1, 2)
  )
tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['Notes_Normalized'])


# - Parameter min_df=2 menghilangkan kata yang sangat jarang (muncul di kurang dari 2 dokumen).
# - Parameter max_df=0.9 menghilangkan kata yang terlalu umum (muncul di lebih dari 90% dokumen).
# - Parameter max_features=1000 membatasi jumlah fitur yang diambil menjadi 1000 kata paling penting.
# - Parameter ngram_range=(1, 2) memungkinkan penangkapan unigram (kata tunggal) dan bigram (2 kata berurutan).

# ### Cosine Similarity

# In[20]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Shape dari Cosine Similarity Matrix: {cosine_sim.shape}")


# ### Sistem Rekomendasi

# In[21]:


def recommend_perfume(name, top_n=5):
    # Cari index parfum di df_clean yang digunakan saat modelling
    idx = df_clean[df_clean['Name'].str.lower() == name.lower()].index
    if len(idx) == 0:
        return f"Parfum '{name}' tidak ditemukan."
    idx = idx[0]

    # Ambil skor similarity dan sortir
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # Lewati dirinya sendiri

    # Ambil nama-nama parfum yang mirip dan similarity scores
    perfume_indices = [i[0] for i in sim_scores]
    return df_clean[['Name', 'Brand', 'Notes']].iloc[perfume_indices], [i[1] for i in sim_scores]


# ### Top 10 Rekomendasi

# In[22]:


name = "Tihota Eau de Parfum"
brand = df[df['Name'].str.lower() == name.lower()]['Brand'].values[0]
notes = df[df['Name'].str.lower() == name.lower()]['Notes'].values[0]
recommendations, scores = recommend_perfume(name, top_n=10)

# gabungkan hasil rekomendasi dengan skor similarity
recommendations['Similarity Score'] = scores

print(f"\nRekomendasi parfum mirip dengan: {name} ({brand}) - {notes}")
recommendations


# Pada contoh ini, sistem merekomendasikan 10 parfum teratas yang paling mirip dengan parfum "Tihota Eau de Parfum". Hasil rekomendasi menunjukkan bahwa parfum-parfum yang direkomendasikan memiliki kesamaan yang tinggi dengan parfum input, berdasarkan karakteristik aroma mereka.

# # Evaluation

# In[23]:


def evaluate_similarity_scores(scores):
    metrics = {
        'rata_kemiripan': np.mean(scores),
        'min_kemiripan': np.min(scores),
        'max_kemiripan': np.max(scores),
        'rentang_kemiripan': np.max(scores) - np.min(scores),
        'varians_kemiripan': np.var(scores)
    }
    return metrics

eval_metrics = evaluate_similarity_scores(scores)
print("Metrik Kualitas Rekomendasi:")
for metric, value in eval_metrics.items():
    print(f"{metric}: {value:.4f}")


# In[24]:


def analyze_recommendation_diversity(recommendations):
    unique_brands = recommendations['Brand'].nunique()
    brand_counts = recommendations['Brand'].value_counts()

    all_notes = []
    for notes_str in recommendations['Notes']:
        if isinstance(notes_str, str):
            note_list = [n.strip().lower() for n in notes_str.split(',')]
            all_notes.extend(note_list)

    unique_notes = len(set(all_notes))

    metrics = {
        'merek_unik': unique_brands,
        'rasio_keberagaman_merek': unique_brands / len(recommendations),
        'merek_unik': unique_notes,
        'rata_rata_catatan_per_rekomendasi': len(all_notes) / len(recommendations) if len(recommendations) > 0 else 0
    }

    return metrics, brand_counts

diversity_metrics, brand_dist = analyze_recommendation_diversity(recommendations)

print("\nAnalisis Keberagaman Rekomendasi:")
for metric, value in diversity_metrics.items():
    print(f"{metric}: {value:.2f}" if isinstance(value, float) else f"{metric}: {value}")

print("\nDistribusi Merek dalam Rekomendasi:")
print(brand_dist)


# In[25]:


def summarize_model_performance(name, brand, notes, recommendations, scores):
    print("=" * 50)
    print(f"RINGKASAN EVALUASI MODEL")
    print("=" * 50)
    print(f"Parfum yang Dicari: {name}")
    print(f"Merek: {brand}")
    print(f"Aroma: {notes}")
    print("-" * 50)

    sim_metrics = evaluate_similarity_scores(scores)
    diversity_metrics, _ = analyze_recommendation_diversity(recommendations)

    print("METRIK KUALITAS REKOMENDASI")
    print(f"Rata-rata skor kemiripan: {sim_metrics['rata_kemiripan']:.4f}")
    print(f"Rentang skor kemiripan: [{sim_metrics['min_kemiripan']:.4f}, {sim_metrics['max_kemiripan']:.4f}]")
    print(f"Variansi skor kemiripan: {sim_metrics['varians_kemiripan']:.4f}")

    print("\nMETRIK KERAGAMAN REKOMENDASI")
    print(f"Jumlah merek unik: {diversity_metrics['merek_unik']} dari {len(recommendations)}")
    print(f"Rasio keragaman merek: {diversity_metrics['rasio_keberagaman_merek']:.2f}")
    print(f"Jumlah aroma unik: {diversity_metrics['merek_unik']}")
    print(f"Rata-rata aroma per rekomendasi: {diversity_metrics['rata_rata_catatan_per_rekomendasi']:.2f}")

    print("\nREKOMENDASI DENGAN KEMIRIPAN TERTINGGI & TERENDAH")
    highest_idx = np.argmax(scores)
    lowest_idx = np.argmin(scores)
    print(f"Paling mirip: {recommendations.iloc[highest_idx]['Name']} ({scores[highest_idx]:.4f})")
    print(f"Paling tidak mirip: {recommendations.iloc[lowest_idx]['Name']} ({scores[lowest_idx]:.4f})")

    print("=" * 50)

summarize_model_performance(name, brand, notes, recommendations, scores)


# In[ ]:




