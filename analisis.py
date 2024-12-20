from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, to_date, year, month, weekofyear, dayofmonth
import pandas as pd

#=== Inisialisasi SparkSession ===
spark = SparkSession.builder \
    .appName("Analisis Data Retail") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()


#=== Baca Data dari Delta Lake ===
delta_path = "hdfs://localhost:9000/dataMDIK/processed_data"
print("Membaca data dari Delta Lake...")
data = spark.read.format("parquet").load(delta_path)
print("Skema Data: ")
data.printSchema()
data.show(5)

#=== Transformasi Datatype ===
df_trans = data.withColumn('Order_Date', to_date(col('Order_Date'), 'dd/MM/yyyy'))
df_trans = df_trans.withColumn('Ship_Date', to_date(col('Ship_Date'), 'dd/MM/yyyy'))
print("Skema Setelah Konversi: ")
df_trans.printSchema()

#=== Transformasi ke Tabel Dimensi ===
#Dimensi Pelanggan
cust_dim = df_trans.select("Customer_ID", "Customer_Name", "Segment").distinct()

#Dimensi Produk
product_dim = df_trans.select("Product_ID", "Product_Name", "Category", "Sub-Category").distinct()

#Dimensi Lokasi
loc_dim = df_trans.select("Postal_Code", "Country", "Region", "State", "City").distinct()

#Dimensi Waktu
time_dim = df_trans.select("Order_Date").distinct()
time_dim = time_dim.withColumn("Year", year(col("Order_Date"))) \
    .withColumn("Quarter", expr("FLOOR((MONTH(Order_Date) - 1) / 3) + 1")) \
    .withColumn("Month", month(col("Order_Date"))) \
    .withColumn("Week", weekofyear(col("Order_Date"))) \
    .withColumn("Day", dayofmonth(col("Order_Date")))


#=== Transformasi ke Tabel Fakta ===
fact_trans = df_trans.select("Order_ID", "Customer_ID", "Product_ID", "Postal_Code", "Order_Date", "Sales")

#=== Koneksi MySQL ===
#Properti
mysql_url = "jdbc:mysql://localhost:3306/mdik_retail"
tabel = "data_retail"
conn_prop = {
    "user" : "root",
    "password" : "",
    "driver" : "com.mysql.cj.jdbc.Driver"
}

#Simpan Data ke MySQL
def save_to_mysql(data, table_name):
    data.write.jdbc(url=mysql_url, table=table_name, mode='overwrite', properties=conn_prop)
    print(f"Data tabel '{table_name}' berhasil disimpan ke MySQL.")

save_to_mysql(df_trans, "Retail_Data")
save_to_mysql(cust_dim, "Cust_Dim")
save_to_mysql(product_dim, "Product_Dim")
save_to_mysql(loc_dim, "Loc_Dim")
save_to_mysql(time_dim, "Time_Dim")
save_to_mysql(fact_trans, "Fact_Trans")
print("Semua data berhasil diproses dan disimpan.")

#Cek Data di MySQL
def chect_data(data, table_name):
    data = spark.read.jdbc(url = mysql_url, table = table_name, properties = conn_prop)
    data.show(5)
    
print("Cek Data di MySQL:")
chect_data(cust_dim, "Cust_Dim")
chect_data(product_dim, "Product_Dim")
chect_data(loc_dim, "Loc_Dim")
chect_data(time_dim, "Time_Dim")
chect_data(fact_trans, "Fact_Trans")

# === Eksplorasi Data ===
data_pd = df_trans.toPandas()
print("Informasi Mengenai Dataset:")
print(data_pd.info())
print("Statistik Deskriptif (Numerik):")
print(data_pd.describe())
print("Statistik Deskriptif Kategorikal:")
print(data_pd.describe(include=['object']))

#Cek Missing Value
miss_val = data_pd.isnull().sum()
miss_perc = (miss_val/len(data_pd)) * 100
print(pd.DataFrame({'Missing Values': miss_val, 'Percentage': miss_perc}))


# === Analisis Data Menggunakan Query SQL ===
df_trans.createOrReplaceTempView("df_trans")
print()
print("====================ANALISIS DATA MENGGUNAKAN QUERY SQL====================")

# 1. Produk terlaris berdasarkan kategori/sub-kategori
print("Produk Terlaris Berdasarkan Kategori")
query_1 = spark.sql("""
                    SELECT Category, Product_Name, SUM(Sales) AS Total_Sales
                    FROM df_trans 
                    GROUP BY Category, Product_Name 
                    ORDER BY Total_Sales DESC 
                    LIMIT 10""")
query_1.show()
print("Produk Terlaris Berdasarkan Sub-Kategori")
query_2 = spark.sql("""
                    SELECT `Sub-Category`, Product_Name, SUM(Sales) AS Total_Sales
                    FROM df_trans 
                    GROUP BY `Sub-Category`, Product_Name 
                    ORDER BY Total_Sales DESC 
                    LIMIT 10""")
query_2.show()

# 2. Segmentasi pelanggan berdasarkan lokasi geografis dan total belanja
query_3 = spark.sql("""
                    SELECT Country, City, State, Region, SUM(Sales) AS Total_Sales
                    FROM df_trans
                    GROUP BY Country, City, State, Region
                    ORDER BY Total_Sales DESC
                    LIMIT 10
                    """)
print("Segmentasi Pelanggan Berdasarkan Lokasi Geografis dan Total Belanja:")
query_3.show()

# 3. Pola penjualan bulanan/kuartalan/tahunan
#query_4 = spark.sql("""
#            SELECT YEAR(tanggal_penjualan) AS tahun, 
#                MONTH(tanggal_penjualan) AS bulan, 
#                QUARTER(tanggal_penjualan) AS kuartal, 
#                SUM(total_belanja) AS total_penjualan
#            FROM data_retail
#            GROUP BY tahun, bulan, kuartal
#            ORDER BY tahun, bulan
#            """)
#print("Pola Penjualan Bulanan/Kuartalan/Tahunan:")
#query_4.show()

# 4. Perbandingan penjualan antar-lokasi
query_5 = spark.sql("""
                    SELECT Region, SUM(Sales) AS Total_Sales
                    FROM df_trans
                    GROUP BY Region
                    ORDER BY Total_Sales DESC
                    LIMIT 10
                    """)
print("Perbandingan Penjualan Antar-Lokasi (Region):")
query_5.show()

query_6 = spark.sql("""
                    SELECT Country, SUM(Sales) AS Total_Sales
                    FROM df_trans
                    GROUP BY Country
                    ORDER BY Total_Sales DESC
                    LIMIT 10
                    """)
print("Perbandingan Penjualan Antar-Lokasi (Country):")
query_6.show()