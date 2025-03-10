import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# Set page config
st.set_page_config(layout="wide", page_title="Dashboard Penyewaan Sepeda")

# Set tema warna kustom
custom_colors = ["#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0", "#118AB2", "#073B4C"]
custom_palette = sns.color_palette(custom_colors)
sns.set_palette(custom_palette)
sns.set(style='darkgrid')

# Menyiapkan data day_df
day_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "day.csv"))

# Menghapus kolom yang tidak diperlukan
drop_col = ['windspeed']
day_df = day_df.drop(columns=drop_col, errors='ignore')

# Mengubah nama judul kolom
day_df.rename(columns={
    'dteday': 'tanggal',
    'yr': 'tahun',
    'mnth': 'bulan',
    'weathersit': 'kondisi_cuaca',
    'cnt': 'jumlah',
    'casual': 'pengguna_kasual',
    'registered': 'pengguna_terdaftar',
    'temp': 'suhu',
    'atemp': 'suhu_terasa',
    'hum': 'kelembaban',
    'season': 'musim',
    'holiday': 'hari_libur',
    'weekday': 'hari',
    'workingday': 'hari_kerja'
}, inplace=True)

# Mengubah format tanggal
day_df['tanggal'] = pd.to_datetime(day_df['tanggal'])

# Mengubah angka menjadi keterangan
day_df['bulan'] = day_df['bulan'].map({
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
    7: 'Jul', 8: 'Agu', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'
})

day_df['hari'] = day_df['hari'].map({
    0: 'Minggu', 1: 'Senin', 2: 'Selasa', 3: 'Rabu', 4: 'Kamis', 5: 'Jumat', 6: 'Sabtu'
})

day_df['hari_libur'] = day_df['hari_libur'].map({
    0: 'Bukan Hari Libur',
    1: 'Hari Libur'
})

day_df['hari_kerja'] = day_df['hari_kerja'].map({
    0: 'Akhir Pekan/Libur',
    1: 'Hari Kerja'
})

# Mengubah angka musim menjadi nama musim
day_df['musim'] = day_df['musim'].map({
    1: 'Musim Semi',
    2: 'Musim Panas',
    3: 'Musim Gugur',
    4: 'Musim Dingin'
})

# Mengubah angka kondisi cuaca menjadi keterangan
day_df['kondisi_cuaca'] = day_df['kondisi_cuaca'].map({
    1: 'Cerah',
    2: 'Berkabut/Berawan',
    3: 'Hujan Ringan',
    4: 'Hujan Lebat'
})

# Normalisasi suhu, kelembaban (skala 0-1 menjadi persentase)
day_df['suhu'] = day_df['suhu'] * 100
day_df['kelembaban'] = day_df['kelembaban'] * 100

# Menyiapkan berbagai jenis dataframe
def create_daily_rent_df(df):
    daily_rent_df = df.groupby(by='tanggal').agg({
        'jumlah': 'sum'
    }).reset_index()
    return daily_rent_df

def create_daily_casual_rent_df(df):
    daily_casual_rent_df = df.groupby(by='tanggal').agg({
        'pengguna_kasual': 'sum'
    }).reset_index()
    return daily_casual_rent_df

def create_daily_registered_rent_df(df):
    daily_registered_rent_df = df.groupby(by='tanggal').agg({
        'pengguna_terdaftar': 'sum'
    }).reset_index()
    return daily_registered_rent_df

def create_monthly_rent_df(df):
    monthly_rent_df = df.groupby(by='bulan').agg({
        'jumlah': 'sum',
        'pengguna_kasual': 'sum',
        'pengguna_terdaftar': 'sum'
    })
    ordered_months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
        'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'
    ]
    monthly_rent_df = monthly_rent_df.reindex(ordered_months, fill_value=0)
    return monthly_rent_df

def create_weekday_rent_df(df):
    weekday_rent_df = df.groupby(by='hari').agg({
        'jumlah': 'sum',
        'pengguna_kasual': 'sum',
        'pengguna_terdaftar': 'sum'
    }).reset_index()
    # Mengurutkan berdasarkan hari dalam seminggu
    correct_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    weekday_rent_df['hari'] = pd.Categorical(weekday_rent_df['hari'], categories=correct_order, ordered=True)
    weekday_rent_df = weekday_rent_df.sort_values('hari')
    return weekday_rent_df

def create_workingday_rent_df(df):
    workingday_rent_df = df.groupby(by='hari_kerja').agg({
        'jumlah': 'sum',
        'pengguna_kasual': 'sum',
        'pengguna_terdaftar': 'sum'
    }).reset_index()
    return workingday_rent_df

def create_holiday_rent_df(df):
    holiday_rent_df = df.groupby(by='hari_libur').agg({
        'jumlah': 'sum',
        'pengguna_kasual': 'sum',
        'pengguna_terdaftar': 'sum'
    }).reset_index()
    return holiday_rent_df

def create_season_rent_df(df):
    season_rent_df = df.groupby(by='musim').agg({
        'jumlah': 'sum',
        'pengguna_kasual': 'sum',
        'pengguna_terdaftar': 'sum'
    }).reset_index()
    # Mengurutkan berdasarkan urutan musim
    correct_order = ['Musim Semi', 'Musim Panas', 'Musim Gugur', 'Musim Dingin']
    season_rent_df['musim'] = pd.Categorical(season_rent_df['musim'], categories=correct_order, ordered=True)
    season_rent_df = season_rent_df.sort_values('musim')
    return season_rent_df

def create_weather_rent_df(df):
    weather_rent_df = df.groupby(by='kondisi_cuaca').agg({
        'jumlah': 'sum',
        'pengguna_kasual': 'sum',
        'pengguna_terdaftar': 'sum'
    }).reset_index()
    return weather_rent_df

def create_temp_humidity_df(df):
    # Membuat pengelompokan berdasarkan rentang suhu dan kelembaban
    temp_humidity_df = df.copy()
    temp_humidity_df['rentang_suhu'] = pd.cut(temp_humidity_df['suhu'], bins=5, labels=["Sangat Dingin", "Dingin", "Sedang", "Hangat", "Panas"])
    temp_humidity_df['rentang_kelembaban'] = pd.cut(temp_humidity_df['kelembaban'], bins=5, labels=["Sangat Kering", "Kering", "Normal", "Lembab", "Sangat Lembab"])
    
    # Agregasi berdasarkan rentang
    temp_agg = temp_humidity_df.groupby('rentang_suhu')['jumlah'].mean().reset_index()
    humidity_agg = temp_humidity_df.groupby('rentang_kelembaban')['jumlah'].mean().reset_index()
    
    return temp_agg, humidity_agg

def create_hourly_pattern_df(df):
    # Untuk simulasi pola per jam (karena data asli adalah harian)
    # Membuat distribusi penyewaan berdasarkan hari kerja vs akhir pekan
    weekday_pattern = np.array([2, 5, 15, 10, 7, 12, 15, 20, 10, 8, 12, 15, 20, 18, 15, 25, 30, 35, 20, 15, 10, 8, 5, 3])
    weekend_pattern = np.array([1, 2, 3, 2, 1, 2, 5, 10, 15, 20, 25, 30, 35, 30, 25, 30, 35, 30, 25, 20, 15, 10, 5, 2])
    
    hourly_data = []
    for hour in range(24):
        weekday_count = int(df[df['hari_kerja'] == 'Hari Kerja']['jumlah'].sum() * weekday_pattern[hour] / weekday_pattern.sum())
        weekend_count = int(df[df['hari_kerja'] == 'Akhir Pekan/Libur']['jumlah'].sum() * weekend_pattern[hour] / weekend_pattern.sum())
        
        hourly_data.append({
            'jam': f"{hour:02d}:00",
            'hari_kerja': weekday_count,
            'akhir_pekan': weekend_count
        })
    
    return pd.DataFrame(hourly_data)

# Membuat Dashboard Streamlit
st.title('Dashboard Penyewaan Sepeda')

# Membuat komponen filter
min_date = day_df['tanggal'].min().date()
max_date = day_df['tanggal'].max().date()

with st.sidebar:
    st.header('Filter')
    
    # Tanggal
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    
    # Tipe hari
    day_type = st.radio(
        'Tipe Hari',
        options=['Semua', 'Hari Kerja', 'Akhir Pekan/Libur']
    )
    
    # Filter musim
    available_seasons = day_df['musim'].unique().tolist()
    selected_seasons = st.multiselect('Musim', available_seasons, default=available_seasons)
    
    # Filter kondisi cuaca
    available_weather = day_df['kondisi_cuaca'].unique().tolist()
    selected_weather = st.multiselect('Kondisi Cuaca', available_weather, default=available_weather)

# Filter data berdasarkan seleksi
main_df = day_df[(day_df['tanggal'].dt.date >= start_date) & 
                 (day_df['tanggal'].dt.date <= end_date)]

if day_type == 'Hari Kerja':
    main_df = main_df[main_df['hari_kerja'] == 'Hari Kerja']
elif day_type == 'Akhir Pekan/Libur':
    main_df = main_df[main_df['hari_kerja'] == 'Akhir Pekan/Libur']

# Filter berdasarkan musim dan cuaca
if selected_seasons:
    main_df = main_df[main_df['musim'].isin(selected_seasons)]
if selected_weather:
    main_df = main_df[main_df['kondisi_cuaca'].isin(selected_weather)]

# Menyiapkan berbagai dataframe dari data yang sudah difilter
daily_rent_df = create_daily_rent_df(main_df)
daily_casual_rent_df = create_daily_casual_rent_df(main_df)
daily_registered_rent_df = create_daily_registered_rent_df(main_df)
monthly_rent_df = create_monthly_rent_df(main_df)
weekday_rent_df = create_weekday_rent_df(main_df)
workingday_rent_df = create_workingday_rent_df(main_df)
holiday_rent_df = create_holiday_rent_df(main_df)
season_rent_df = create_season_rent_df(main_df)
weather_rent_df = create_weather_rent_df(main_df)
temp_agg, humidity_agg = create_temp_humidity_df(main_df)
hourly_pattern_df = create_hourly_pattern_df(main_df)

# Membuat jumlah penyewaan harian
st.header('Ringkasan Penyewaan')
col1, col2, col3 = st.columns(3)

with col1:
    daily_rent_casual = daily_casual_rent_df['pengguna_kasual'].sum()
    st.metric('Pengguna Kasual', value=f"{daily_rent_casual:,.0f}")

with col2:
    daily_rent_registered = daily_registered_rent_df['pengguna_terdaftar'].sum()
    st.metric('Pengguna Terdaftar', value=f"{daily_rent_registered:,.0f}")
 
with col3:
    daily_rent_total = daily_rent_df['jumlah'].sum()
    st.metric('Total Pengguna', value=f"{daily_rent_total:,.0f}")

# Pola Penyewaan per Jam
st.header('Pola Penyewaan per Jam (Estimasi)')
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    hourly_pattern_df['jam'],
    hourly_pattern_df['hari_kerja'],
    marker='o',
    linewidth=2,
    color=custom_colors[0],
    label='Hari Kerja'
)

ax.plot(
    hourly_pattern_df['jam'],
    hourly_pattern_df['akhir_pekan'],
    marker='o',
    linewidth=2,
    color=custom_colors[1],
    label='Akhir Pekan'
)

plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.title('Pola Penyewaan per Jam', fontsize=14)
plt.tight_layout()
st.pyplot(fig)

# Layout dengan kolom
col1, col2 = st.columns(2)

with col1:
    # Membuat jumlah penyewaan bulanan
    st.header('Penyewaan Bulanan')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    monthly_data = monthly_rent_df.reset_index()
    
    ax.bar(
        monthly_data['bulan'],
        monthly_data['pengguna_kasual'],
        color=custom_colors[0],
        label='Pengguna Kasual'
    )
    
    ax.bar(
        monthly_data['bulan'],
        monthly_data['pengguna_terdaftar'],
        bottom=monthly_data['pengguna_kasual'],
        color=custom_colors[1],
        label='Pengguna Terdaftar'
    )
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.title('Penyewaan Berdasarkan Bulan', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Membuat jumlah penyewaan berdasarkan hari
    st.header('Penyewaan Berdasarkan Hari')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Membuat diagram lingkaran
    weekday_totals = weekday_rent_df.set_index('hari')['jumlah']
    
    if not weekday_totals.empty:
        # Create explode list based on the number of slices
        explode = [0.05] * len(weekday_totals)
        
        plt.pie(
            weekday_totals, 
            labels=weekday_totals.index, 
            autopct='%1.1f%%',
            colors=custom_colors[:len(weekday_totals)],
            startangle=90,
            shadow=True,
            explode=explode
        )
        
        plt.title('Distribusi Penyewaan Berdasarkan Hari', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # Clear the display if no slices to plot
        st.write("Tidak ada data untuk ditampilkan.")

# FITUR BARU: Penyewaan Berdasarkan Musim
st.header('Penyewaan Berdasarkan Musim')
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    season_data = season_rent_df
    
    ax.bar(
        season_data['musim'],
        season_data['pengguna_kasual'],
        color=custom_colors[0],
        label='Pengguna Kasual'
    )
    
    ax.bar(
        season_data['musim'],
        season_data['pengguna_terdaftar'],
        bottom=season_data['pengguna_kasual'],
        color=custom_colors[1],
        label='Pengguna Terdaftar'
    )
    
    for i, row in enumerate(season_data['jumlah']):
        ax.text(i, row/2, f"{row:,.0f}", ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Jumlah Penyewaan Berdasarkan Musim', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Persentase untuk masing-masing musim
    season_pct = season_data.set_index('musim')['jumlah'] / season_data['jumlah'].sum() * 100
    st.dataframe(
        pd.DataFrame({
            'Musim': season_pct.index,
            'Persentase (%)': season_pct.values.round(1)
        }).set_index('Musim'),
        use_container_width=True
    )

# FITUR BARU: Penyewaan Berdasarkan Kondisi Cuaca
st.header('Penyewaan Berdasarkan Kondisi Cuaca')
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weather_data = weather_rent_df
    
    ax.bar(
        weather_data['kondisi_cuaca'],
        weather_data['pengguna_kasual'],
        color=custom_colors[0],
        label='Pengguna Kasual'
    )
    
    ax.bar(
        weather_data['kondisi_cuaca'],
        weather_data['pengguna_terdaftar'],
        bottom=weather_data['pengguna_kasual'],
        color=custom_colors[1],
        label='Pengguna Terdaftar'
    )
    
    for i, row in enumerate(weather_data['jumlah']):
        ax.text(i, row/2, f"{row:,.0f}", ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Jumlah Penyewaan Berdasarkan Kondisi Cuaca', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Persentase untuk masing-masing kondisi cuaca
    weather_pct = weather_data.set_index('kondisi_cuaca')['jumlah'] / weather_data['jumlah'].sum() * 100
    st.dataframe(
        pd.DataFrame({
            'Kondisi Cuaca': weather_pct.index,
            'Persentase (%)': weather_pct.values.round(1)
        }).set_index('Kondisi Cuaca'),
        use_container_width=True
    )

# Membuat jumlah penyewaan berdasarkan hari libur
st.header('Penyewaan Berdasarkan Hari Libur')

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    holiday_data = holiday_rent_df
    
    ax.bar(
        holiday_data['hari_libur'],
        holiday_data['pengguna_kasual'],
        label='Pengguna Kasual',
        color=custom_colors[0]
    )
    
    ax.bar(
        holiday_data['hari_libur'],
        holiday_data['pengguna_terdaftar'],
        bottom=holiday_data['pengguna_kasual'],
        label='Pengguna Terdaftar',
        color=custom_colors[1]
    )
    
    for i, row in enumerate(holiday_data['jumlah']):
        ax.text(i, row/2, f"{row:,.0f}", ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    ax.set_xlabel(None)
    ax.set_ylabel('Jumlah Penyewaan')
    ax.tick_params(axis='x', labelsize=10, rotation=30)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Persentase untuk masing-masing kondisi cuaca
    holiday_pct = holiday_data.set_index('hari_libur')['jumlah'] / holiday_data['jumlah'].sum() * 100
    st.dataframe(
        pd.DataFrame({
            'Hari Libur': holiday_pct.index,
            'Persentase (%)': holiday_pct.values.round(1)
        }).set_index('Hari Libur'),
        use_container_width=True
    )

# Visualisasi Tambahan - Pengaruh Suhu dan Kelembaban
st.header('Pengaruh Suhu dan Kelembaban')

col1, col2 = st.columns(2)

with col1:
    # Visualisasi pengaruh suhu
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        x='rentang_suhu',
        y='jumlah',
        data=temp_agg,
        palette=[custom_colors[0], custom_colors[1], custom_colors[2], custom_colors[3], custom_colors[4]],
        ax=ax
    )
    
    plt.title('Rata-rata Penyewaan Berdasarkan Suhu')
    plt.ylabel('Rata-rata Penyewaan')
    plt.xlabel('Rentang Suhu')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Visualisasi pengaruh kelembaban
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        x='rentang_kelembaban',
        y='jumlah',
        data=humidity_agg,
        palette=[custom_colors[0], custom_colors[1], custom_colors[2], custom_colors[3], custom_colors[4]],
        ax=ax
    )
    
    plt.title('Rata-rata Penyewaan Berdasarkan Kelembaban')
    plt.ylabel('Rata-rata Penyewaan')
    plt.xlabel('Rentang Kelembaban')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# FITUR BARU: Korelasi Musim dengan Penyewaan
st.header('Korelasi Musim, Cuaca, dan Penyewaan')

# Buat visualisasi heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Pivot tabel untuk membuat data yang sesuai untuk heatmap
if not main_df.empty and len(main_df['musim'].unique()) > 1 and len(main_df['kondisi_cuaca'].unique()) > 1:
    heatmap_data = main_df.pivot_table(
        index='musim', 
        columns='kondisi_cuaca', 
        values='jumlah', 
        aggfunc='mean'
    )
    
    # Membuat custom colormap dari biru muda ke biru tua
    cmap = LinearSegmentedColormap.from_list('custom_blue', ['#DBEDFF', '#0068C9'])
    
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".0f", 
        linewidths=.5, 
        ax=ax,
        cmap=cmap
    )
    
    plt.title('Rata-rata Penyewaan Berdasarkan Musim dan Kondisi Cuaca')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("Data tidak cukup untuk menampilkan heatmap korelasi.")