import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="E-Commerce Analysis Dashboard",
    layout="wide"
)

# Menambahkan judul dashboard
st.title("E-Commerce Analysis Dashboard")

# Fungsi untuk memuat data dengan caching
@st.cache_data
def load_data():
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path to the CSV file
    file_path = os.path.join(current_dir, "clean_data.csv")
    
    # Load the data
    final_df = pd.read_csv(file_path)
    
    # Convert date columns
    date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_columns:
        if col in final_df.columns and not pd.api.types.is_datetime64_any_dtype(final_df[col]):
            final_df[col] = pd.to_datetime(final_df[col])
    
    return final_df

# Memuat data
try:
    final_df = load_data()
    
    # ===== VISUALISASI METRIK DISTRIBUSI =====
    st.subheader("Metrik Distribusi Data")
    col1, col2 = st.columns(2)
    
    # Visualisasi jumlah penjual berdasarkan kota
    with col1:
        st.subheader("Persebaran Penjual Di Setiap Kota")
        sellers_by_city = final_df.groupby(by="seller_city").seller_id.nunique().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sellers_by_city.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Menampilkan deskripsi insight
        st.markdown("""
        **Insight:**
        Kota dengan penjual terbanyak adalah sao paulo sebanyak 694 yang berbeda cukup jauh dengan kota dibawahnya 127. 
        Hal ini terlihat bahwa penjual terpusat di kota sao paulo.
        """)
    
    # Visualisasi jumlah pelanggan berdasarkan kota
    with col2:
        st.subheader("Persebaran Pembeli Di Setiap Kota")
        customers_by_city = final_df.groupby(by="customer_city").customer_unique_id.nunique().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        customers_by_city.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
        # Menampilkan deskripsi insight
        st.markdown("""
        **Insight:**
        Kota dengan pembeli paling banyak adalah di kota sao paulo yang juga sama dengan kota yang memiliki penjual terbanyak.
        """)
    
    col3, col4 = st.columns(2)
    
    # Visualisasi distribusi metode pembayaran
    with col3:
        st.subheader("Persebaran Penggunaan Jenis Pembayaran")
        payment_types = final_df.payment_type.value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        payment_types.plot(kind='bar', ax=ax)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Menampilkan deskripsi insight
        st.markdown("""
        **Insight:**
        Jenis pembayaran yang paling sering digunakan pada aplikasi e-commerce adalah kartu kredit
        """)
    
    # Visualisasi distribusi skor review
    with col4:
        st.subheader("Persebaran Skor Review Transaksi Produk")
        review_scores = final_df.review_score.value_counts().sort_index(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        review_scores.plot(kind='bar', ax=ax)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Menampilkan deskripsi insight
        st.markdown("""
        **Insight:**
        Banyak pembeli yang puas dengan barang yang mereka beli, tetapi di sisi lain tidak sedikit juga dari mereka yang tidak puas
        """)
        
    # ===== VISUALISASI KATEGORI PRODUK DENGAN PENJUALAN TERTINGGI =====
    st.subheader("Distribusi Penjualan Berdasarkan Kategori Produk")
    
    # Menghitung penjualan berdasarkan kategori
    if 'product_category_name_english' in final_df.columns and 'payment_value' in final_df.columns:
        # Mengelompokkan data berdasarkan kategori produk
        sales_by_category = final_df.groupby('product_category_name_english').agg({
            'payment_value': 'sum',  # Menghitung total nilai pembayaran
            'order_id': 'nunique'    # Menghitung jumlah pesanan unik
        }).reset_index()
        
        # Mengurutkan berdasarkan total penjualan
        sales_by_category = sales_by_category.sort_values('payment_value', ascending=False)
        
        # Mengambil 15 kategori teratas
        top_categories = sales_by_category.head(15)
        
        # Membuat visualisasi
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Membuat bar plot horizontal dengan warna gradien
        bars = ax.barh(top_categories['product_category_name_english'],
                      top_categories['payment_value'],
                      color=plt.cm.viridis(np.linspace(0, 0.8, len(top_categories))))
        
        # Menambahkan anotasi nilai pada setiap bar
        for i, bar in enumerate(bars):
            value = top_categories['payment_value'].iloc[i]
            ax.text(value + (sales_by_category['payment_value'].max() * 0.01),
                   bar.get_y() + bar.get_height()/2,
                   f'$ {value:,.2f}',
                   va='center',
                   fontweight='bold')
        
        ax.set_title('15 Product Categories with Highest Sales', fontsize=16, pad=20)
        ax.set_xlabel('Total Sales ($)', fontsize=12)
        ax.set_ylabel('Product Category', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Menampilkan deskripsi insight
        st.markdown("""
        **Insight:**
        * Kategori bed_bath_table memimpin dengan penjualan tertinggi (2,16 juta), jauh di atas kategori lainnya
        * Produk kebutuhan rumah tangga mendominasi penjualan teratas, dengan health_beauty (1,66 juta) dan computers_accessories (1,59 juta) melengkapi tiga besar.
        * Terdapat kesenjangan besar antara kategori teratas dan terbawah, dengan telephony hanya mencapai sekitar 487 ribu.
        * Konsumen lebih banyak membelanjakan uang untuk kebutuhan rumah tangga, kesehatan, dan teknologi dibanding kategori lainnya.
        """)
    else:
        st.warning("Kolom yang diperlukan tidak ditemukan. Pastikan 'product_category_name_english' dan 'payment_value' tersedia.")
        
    # ===== VISUALISASI TREN PENJUALAN BULANAN =====
    st.subheader("Tren Penjualan Kategori Produk Unggulan Seiring Waktu")
    
    # Menambahkan kolom bulan-tahun untuk pengelompokan
    final_df['year_month'] = final_df['order_purchase_timestamp'].dt.strftime('%Y-%m')
    
    # Menghitung total penjualan per kategori
    total_sales_by_category = final_df.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False)
    
    # Mengambil 5 kategori teratas
    top_5_categories = total_sales_by_category.head(5).index.tolist()
    
    # Memfilter data untuk 5 kategori teratas
    top_categories_data = final_df[final_df['product_category_name_english'].isin(top_5_categories)]
    
    # Menghitung penjualan bulanan untuk 5 kategori teratas
    monthly_sales = top_categories_data.groupby(['year_month', 'product_category_name_english'])['price'].sum().reset_index()
    
    # Pivot data untuk plotting
    pivot_data = monthly_sales.pivot(index='year_month', columns='product_category_name_english', values='price')
    
    # Mengurutkan index berdasarkan tanggal
    pivot_data.index = pd.to_datetime(pivot_data.index, format='%Y-%m')
    pivot_data = pivot_data.sort_index()
    
    # Membuat plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot setiap kategori dengan warna berbeda
    for category in top_5_categories:
        if category in pivot_data.columns:
            ax.plot(pivot_data.index, pivot_data[category], 
                   marker='o', markersize=5, 
                   linewidth=2, 
                   label=category)
    
    ax.set_title('Monthly Sales Trends for Top 5 Product Categories', fontsize=16, pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Sales ($)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Menampilkan deskripsi insight
    st.markdown("""
    **Insight:**
    * Kategori health_beauty menunjukkan pertumbuhan paling konsisten, mencapai penjualan tertinggi sekitar 125.000 di akhir 2018.
    * Watches_gifts menampilkan volatilitas tinggi dengan puncak sekitar 125.000 pada April 2018.
    * Bed_bath_table awalnya memimpin dengan puncak 115.000 pada akhir 2017, namun mengalami penurunan drastis mendekati nol di Oktober 2018.
    * Sports_leisure dan computers_accessories menunjukkan tren penurunan setelah mencapai puncak di awal 2018.
    * Pola musiman terlihat jelas pada semua kategori, dengan lonjakan penjualan di periode-periode tertentu.
    """)
    
    # ===== VISUALISASI BERAT PRODUK VS WAKTU PENGIRIMAN =====
    st.subheader("Pengaruh Karakteristik Produk terhadap Waktu Pengiriman")
    
    # Menghitung waktu pengiriman dalam hari
    final_df['delivery_time_days'] = (
        final_df['order_delivered_customer_date'] -
        final_df['order_purchase_timestamp']
    ).dt.days
    
    # Memfilter data yang valid (waktu pengiriman masuk akal dan memiliki data berat produk)
    valid_data = final_df[
        (final_df['delivery_time_days'] > 0) &
        (final_df['delivery_time_days'] <= final_df['delivery_time_days'].quantile(0.99)) &
        (final_df['product_weight_g'] > 0)
    ].copy()
    
    # Membuat kategori berat produk yang intuitif
    valid_data['weight_category'] = pd.cut(
        valid_data['product_weight_g'],
        bins=[0, 500, 1000, 2000, 5000, 10000, float('inf')],
        labels=['< 0.5 kg', '0.5-1 kg', '1-2 kg', '2-5 kg', '5-10 kg', '> 10 kg']
    )
    
    # Membuat figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Menghitung statistik untuk setiap kategori berat
    weight_stats = valid_data.groupby('weight_category', observed=True).agg({
        'delivery_time_days': ['mean', 'median', 'count', 'std']
    })
    
    # Merapikan nama kolom
    weight_stats.columns = ['mean_days', 'median_days', 'count', 'std_days']
    weight_stats = weight_stats.reset_index()
    
    # Membuat bar chart untuk rata-rata waktu pengiriman per kategori berat
    bar_plot = sns.barplot(
        x='weight_category',
        y='mean_days',
        data=weight_stats,
        palette='YlOrRd',  # Palet warna kuning ke merah
        hue='weight_category',
        legend=False,
        ax=ax
    )
    
    # Menambahkan error bar untuk menunjukkan variabilitas
    ax.errorbar(
        x=weight_stats.index,
        y=weight_stats['mean_days'],
        yerr=weight_stats['std_days'] / np.sqrt(weight_stats['count']),  # Standard error
        fmt='none',
        color='black',
        capsize=5
    )
    
    # Menambahkan label nilai di atas bar
    for i, row in enumerate(weight_stats.itertuples()):
        ax.text(
            i, row.mean_days + 0.5,
            f'{row.mean_days:.1f} days',
            ha='center',
            fontweight='bold'
        )
        # Menambahkan jumlah produk di bawah label kategori
        ax.text(
            i, -1.5,
            f'n = {row.count:,}',
            ha='center',
            fontsize=9
        )
    
    # Menambahkan referensi rata-rata keseluruhan
    overall_mean = valid_data['delivery_time_days'].mean()
    ax.axhline(y=overall_mean, color='navy', linestyle='--', alpha=0.7)
    ax.text(
        len(weight_stats) - 1, overall_mean + 0.7,
        f'Rata-rata keseluruhan: {overall_mean:.1f} hari',
        ha='right',
        color='navy',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    # Menambahkan judul dan label
    ax.set_xlabel('Berat Produk (kg)', fontsize=12)
    ax.set_ylabel('Rata-rata Waktu Pengiriman (hari)', fontsize=12)
    
    # Menghitung korelasi antara berat produk dan waktu pengiriman
    correlation = valid_data[['product_weight_g', 'delivery_time_days']].corr().iloc[0, 1]
    
    # Menambahkan catatan korelasi
    plt.figtext(
        0.02, 0.01,
        f"Korelasi antara berat produk dan waktu pengiriman: {correlation:.3f}",
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Menampilkan deskripsi insight
    st.markdown("""
    **Insight:**
    * Terdapat tren peningkatan waktu pengiriman seiring bertambahnya berat produk, dengan produk terberat (>10 kg) membutuhkan waktu pengiriman terlama yaitu 13.8 hari
    * Produk ringan (0.5 kg) memiliki waktu pengiriman tercepat yaitu 11.0 hari, dengan selisih 2.8 hari dibandingkan produk terberat
    * Peningkatan waktu pengiriman paling signifikan terjadi pada kategori produk berat (5-10 kg dan >10 kg)
    * Meskipun terdapat pola peningkatan, korelasi antara berat produk dan waktu pengiriman tergolong lemah (0.084), menunjukkan bahwa berat produk bukan satu-satunya faktor dominan yang mempengaruhi waktu pengiriman
    * Rata-rata keseluruhan waktu pengiriman adalah 11.6 hari berdasarkan analisis dari 117,972 pesanan yang telah terkirim
    * Terjadi anomali kecil pada kategori 2-5 kg (11.8 hari) yang sedikit lebih cepat dibanding kategori 1-2 kg (11.9 hari)
    """)
    
    # ===== VISUALISASI WAKTU PENGIRIMAN VS SKOR REVIEW =====
    st.subheader("Pengaruh Waktu Pengiriman terhadap Review Score")
    
    # Membuang data pesanan yang tidak memiliki tanggal pengiriman ke pelanggan atau skor review
    delivered_orders = final_df.dropna(subset=['order_delivered_customer_date', 'review_score'])
    delivery_review = delivered_orders.copy()
    
    # Menghitung waktu pengiriman dalam satuan hari
    delivery_review.loc[:, 'delivery_time'] = (
        delivery_review['order_delivered_customer_date'] -
        delivery_review['order_purchase_timestamp']
    ).dt.days
    
    # Memfilter data untuk hanya mengambil waktu pengiriman yang positif
    mask = (delivery_review['delivery_time'] > 0)
    valid_review_data = delivery_review[mask].copy()
    
    # Mengatasi outlier dengan hanya mengambil data sampai persentil ke-99
    max_delivery_time = valid_review_data['delivery_time'].quantile(0.99)
    valid_review_data = valid_review_data[valid_review_data['delivery_time'] <= max_delivery_time].copy()
    
    # Membuat kategori waktu pengiriman untuk mempermudah analisis dan visualisasi
    valid_review_data.loc[:, 'delivery_time_category'] = pd.cut(
        valid_review_data['delivery_time'],
        bins=[0, 3, 7, 14, 21, float('inf')],
        labels=['0-3 days', '4-7 days', '8-14 days', '15-21 days', '>21 days']
    )
    
    # Membuat subplots dengan dua grafik: boxplot di atas dan line plot di bawah
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # Membuat boxplot untuk menunjukkan distribusi review score berdasarkan kategori waktu pengiriman
    sns.boxplot(
        x='delivery_time_category',
        y='review_score',
        hue='delivery_time_category',
        data=valid_review_data,
        palette='viridis',
        legend=False,
        ax=ax1
    )
    
    # Menambahkan judul dan label pada boxplot
    ax1.set_title('Distribusi Review Core berdasarkan Kategori Waktu Pengiriman', fontsize=16, pad=20)
    ax1.set_xlabel('Kategori Waktu Pengiriman', fontsize=12)
    ax1.set_ylabel('Review Score (1-5)', fontsize=12)
    ax1.set_ylim(0.5, 5.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Menghitung rata-rata review score untuk setiap kategori waktu pengiriman
    category_avg = valid_review_data.groupby('delivery_time_category', observed=True)['review_score'].mean()
    
    # Menambahkan label rata-rata review score di atas setiap boxplot
    for i, category in enumerate(category_avg.index):
        ax1.text(
            i,
            5.3,
            f'Rata-rata: {category_avg[category]:.2f}',
            ha='center',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
    
    # Membuat bins untuk waktu pengiriman dengan interval 2 hari untuk analisis tren yang lebih halus
    delivery_bins = list(range(0, int(max_delivery_time) + 1, 2))
    
    # Mengelompokkan data berdasarkan bin waktu pengiriman dan menghitung rata-rata dan jumlah
    review_by_delivery = valid_review_data.groupby(
        pd.cut(valid_review_data['delivery_time'], bins=delivery_bins),
        observed=True
    )['review_score'].agg(['mean', 'count']).reset_index()
    
    # Mengambil nilai tengah dari setiap bin untuk digunakan sebagai nilai x dalam plot
    review_by_delivery['delivery_time_mid'] = review_by_delivery['delivery_time'].apply(lambda x: x.mid)
    
    # Memfilter hanya bin dengan minimal 10 data untuk hasil yang lebih reliabel
    review_by_delivery = review_by_delivery[review_by_delivery['count'] >= 10]
    
    # Membuat line plot untuk menunjukkan tren review score berdasarkan waktu pengiriman
    ax2.plot(
        review_by_delivery['delivery_time_mid'],
        review_by_delivery['mean'],
        'o-',  # Gaya plot: garis dengan marker lingkaran
        color='#3366cc',
        linewidth=2,
        markersize=8
    )
    
    # Menghitung standard error untuk setiap bin untuk visualisasi interval kepercayaan
    review_by_delivery['se'] = valid_review_data.groupby(
        pd.cut(valid_review_data['delivery_time'], bins=delivery_bins),
        observed=True
    )['review_score'].agg(lambda x: x.std() / np.sqrt(len(x))).values
    
    # Menambahkan interval kepercayaan 95% (±1.96 SE) sebagai area berbayang pada line plot
    ax2.fill_between(
        review_by_delivery['delivery_time_mid'],
        review_by_delivery['mean'] - 1.96 * review_by_delivery['se'],
        review_by_delivery['mean'] + 1.96 * review_by_delivery['se'],
        color='#3366cc',
        alpha=0.2
    )
    
    # Menambahkan garis horizontal yang menunjukkan rata-rata keseluruhan review score
    overall_mean = valid_review_data['review_score'].mean()
    ax2.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Rata-rata keseluruhan: {overall_mean:.2f}')
    
    # Menambahkan judul dan label pada line plot
    ax2.set_title('Tren Ewview Score berdasarkan Waktu Pengiriman', fontsize=16, pad=20)
    ax2.set_xlabel('Waktu Pengiriman (hari)', fontsize=12)
    ax2.set_ylabel('Rata-rata Review Score', fontsize=12)
    ax2.set_ylim(3.0, 5.0)  # Mengatur batas y-axis untuk fokus pada variasi yang relevan
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Menghitung korelasi antara waktu pengiriman dan review score
    correlation = valid_review_data[['delivery_time', 'review_score']].corr().iloc[0, 1]
    
    # Menambahkan teks yang menunjukkan nilai korelasi di pojok kiri bawah grafik
    ax2.text(
        0.02, 0.05,
        f'Correlation: {correlation:.3f}',
        transform=ax2.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Menambahkan scatter plot dengan ukuran titik sesuai jumlah pesanan
    sizes = review_by_delivery['count'] / review_by_delivery['count'].max() * 100 + 20
    scatter = ax2.scatter(
        review_by_delivery['delivery_time_mid'],
        review_by_delivery['mean'],
        s=sizes,
        alpha=0.5,
        c=review_by_delivery['count'],
        cmap='viridis',
        edgecolor='black'
    )
    
    # Menambahkan color bar untuk menunjukkan hubungan warna dengan jumlah pesanan
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label('Jumlah Pesanan', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Menampilkan deskripsi insight
    st.markdown("""
    **Insight:**
    * Terdapat hubungan negatif yang jelas antara waktu pengiriman dan kepuasan pelanggan. Semakin lama waktu pengiriman, semakin rendah rating review yang diberikan
    * Pengiriman cepat (0-3 hari) mendapatkan rating tertinggi dengan rata-rata 4.39, sementara pengiriman sangat lambat (>21 hari) mendapatkan rating terendah dengan rata-rata hanya 3.08
    * Variabilitas review meningkat signifikan pada kategori pengiriman yang lebih lama. Untuk pengiriman 0-14 hari, sebagian besar review terkonsentrasi antara 4-5 bintang. Namun untuk pengiriman >21 hari, distribusi melebar hingga ke rating 1 bintang
    * Dari grafik trend di bawahnya, terlihat penurunan rating yang stabil seiring bertambahnya waktu pengiriman, dengan penurunan paling tajam terjadi setelah 15-20 hari
    * Korelasi negatif sebesar -0.271 mengkonfirmasi adanya hubungan yang cukup kuat antara keterlambatan pengiriman dan penurunan kepuasan pelanggan
    * Rata-rata keseluruhan review adalah 4.04, yang menunjukkan bahwa mayoritas pelanggan masih memberikan rating positif selama pengiriman tidak melebihi 14 hari
    """)
    
    # ===== VISUALISASI PENGARUH TARIF PENGIRIMAN TERHADAP KEPUASAN PELANGGAN =====
    st.subheader("Pengaruh Tarif Pengiriman terhadap Review Score")
    
    # Membersihkan data dengan menghapus baris yang memiliki nilai kosong pada kolom freight_value dan review_score
    clean_df = final_df.dropna(subset=['freight_value', 'review_score']).copy()
    
    # Mendefinisikan rentang nilai untuk kategorisasi tarif pengiriman
    bins = [0, 15, 30, 50, 75, 100, float('inf')]
    labels = ["0-15", "15-30", "30-50", "50-75", "75-100", "100+"]
    
    # Membuat kategori tarif pengiriman
    clean_df['freight_category'] = pd.cut(clean_df['freight_value'], bins=bins, labels=labels)
    
    # Menghitung rata-rata review score untuk setiap kategori tarif pengiriman
    avg_ratings = clean_df.groupby('freight_category')['review_score'].mean()
    
    # Menghitung jumlah pesanan untuk setiap kategori tarif pengiriman
    order_counts = clean_df.groupby('freight_category').size()
    
    # Menghitung korelasi Pearson antara nilai tarif pengiriman dan review score
    correlation = stats.pearsonr(clean_df['freight_value'], clean_df['review_score'])[0]
    
    # Membuat figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Menyiapkan data untuk visualisasi bar chart
    categories = labels
    bar_positions = np.arange(len(categories))
    bar_width = 0.6
    
    # Membuat gradasi warna dari merah ke hijau terbalik
    colors = sns.color_palette("RdYlGn_r", len(categories))
    
    # Membuat bar chart
    bars = ax.bar(
        bar_positions,
        avg_ratings,
        bar_width,
        color=colors,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Menambahkan label nilai rata-rata review score di atas setiap bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.05,
            f'{avg_ratings[i]:.1f}',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )
        
        # Menambahkan jumlah pesanan di tengah setiap bar
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height/2,
            f'{order_counts[i]} orders',
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            color='black'
        )
    
    # Menambahkan judul dan label
    ax.set_title('Pengaruh Tarif Pengiriman terhadap kepuasan pelanggan', fontsize=16, pad=20)
    ax.set_xlabel('Tarif Pengiriman (R$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Rata-rata Review Score (1-5)', fontsize=14, labelpad=10)
    
    # Mengatur ticks pada sumbu x
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(categories, fontsize=12)
    
    # Mengatur batas sumbu y dari 0 hingga 5 (rentang review score)
    ax.set_ylim(0, 5)
    # Mengatur format nilai pada sumbu y dengan 1 desimal
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Menambahkan teks yang menampilkan nilai korelasi di pojok kanan atas
    ax.text(
        0.98, 0.98,
        f'Korelasi: {correlation:.2f}',
        transform=ax.transAxes,
        fontsize=12,
        ha='right',
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
    )
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Menampilkan deskripsi insight
    st.markdown("""
    **Insight:**
    * Semakin tinggi tarif pengiriman, semakin rendah tingkat kepuasan pelanggan
    * Pesanan dengan tarif di bawah 30 memiliki rating rata-rata di atas 4
    * Tarif lebih dari 75 menyebabkan penurunan signifikan pada kepuasan
    * Mayoritas pesanan (78%) menggunakan tarif pengiriman di bawah 50
    """)
    
    # ===== RFM ANALYSIS =====
    st.subheader("RFM Analysis")

    try:
        st.markdown("""
        **RFM Analysis** adalah teknik segmentasi pelanggan yang ampuh untuk mengidentifikasi pelanggan yang paling berharga berdasarkan tiga metrik utama:

        * Recency: Seberapa baru pelanggan melakukan pembelian (kebaruan transaksi)
        * Frequency: Seberapa sering pelanggan melakukan pembelian (frekuensi transaksi)
        * Monetary: Seberapa banyak uang yang dibelanjakan oleh pelanggan (nilai moneter)
        Analisis ini membantu bisnis mengembangkan strategi pemasaran yang ditargetkan untuk berbagai segmen pelanggan yang berbeda. Dengan memahami perilaku pembelian pelanggan, bisnis dapat meningkatkan retensi pelanggan, meningkatkan loyalitas, dan mengoptimalkan pengalaman pelanggan.
        """)
        
        with st.spinner("Calculating RFM metrics..."):
            rfm_df = final_df.groupby(by="customer_unique_id", as_index=False).agg({
                "order_purchase_timestamp": "max",
                "order_id": "nunique",
                "payment_value": "sum"
            })
            
            rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
            rfm_df["max_order_timestamp"] = pd.to_datetime(rfm_df["max_order_timestamp"]).dt.date
            recent_date = pd.to_datetime(final_df["order_purchase_timestamp"]).dt.date.max()
            rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
            rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
            st.write("Sampel data RFM (5 baris pertama):")
            st.dataframe(rfm_df.head())
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]
        
        sns.barplot(
            y="recency", 
            x="customer_id",
            data=rfm_df.sort_values(by="recency", ascending=True).head(5),
            palette=colors, 
            ax=ax[0]
        )
        ax[0].set_ylabel("Days since last purchase")
        ax[0].set_xlabel(None)
        ax[0].set_title("By Recency (days)", loc="center", fontsize=14)
        plt.setp(ax[0].get_xticklabels(), rotation=45, ha='right')
        
        sns.barplot(
            y="frequency", 
            x="customer_id",
            data=rfm_df.sort_values(by="frequency", ascending=False).head(5),
            palette=colors, 
            ax=ax[1]
        )
        ax[1].set_ylabel("Number of orders")
        ax[1].set_xlabel(None)
        ax[1].set_title("By Frequency", loc="center", fontsize=14)
        plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right')
        
        sns.barplot(
            y="monetary", 
            x="customer_id",
            data=rfm_df.sort_values(by="monetary", ascending=False).head(5),
            palette=colors, 
            ax=ax[2]
        )
        ax[2].set_ylabel("Total spent ($)")
        ax[2].set_xlabel(None)
        ax[2].set_title("By Monetary", loc="center", fontsize=14)
        plt.setp(ax[2].get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle("Visualisasi RFM", fontsize=16)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        * Recency : Pelanggan teratas baru saja berbelanja (0 hari) dan Terdapat perbedaan besar antara 2 pelanggan teratas dengan 3 pelanggan lainnya (sekitar 13-18 hari)
        * Frequency : Pelanggan teratas melakukan sekitar 17 pembelian dan Pelanggan lainnya memiliki frekuensi yang jauh lebih rendah (7-9 pembelian)
        * Monetary : Pelanggan teratas menghabiskan sekitar 100.000+ (2x lipat dari pelanggan berikutnya) dan Terjadi penurunan bertahap pada nilai belanja pelanggan lainnya
        Tidak ada pelanggan yang unggul di semua dimensi, menunjukkan segmentasi pelanggan yang berbeda berdasarkan perilaku pembelian mereka.
        """)
        
    except Exception as e:
        st.error(f"Error in RFM analysis: {e}")
        st.info("Please ensure your dataset contains the required columns for RFM analysis: customer_unique_id, order_purchase_timestamp, order_id, and payment_value.")
except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.info("Pastikan semua file CSV yang diperlukan berada di direktori yang sama dengan script ini.")