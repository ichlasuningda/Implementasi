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
def check_data(data, table_name):
    data = spark.read.jdbc(url = mysql_url, table = table_name, properties = conn_prop)
    print("Nama Tabel: ", table_name)
    data.show(5)
    data.printSchema()
    
print("Cek Data di MySQL:")
check_data(cust_dim, "Cust_Dim")
check_data(product_dim, "Product_Dim")
check_data(loc_dim, "Loc_Dim")
check_data(time_dim, "Time_Dim")
check_data(fact_trans, "Fact_Trans")


#=== Eksplorasi Tabel Fakta ===
data_eks = fact_trans.toPandas()
print("Informasi Mengenai Dataset Pada Tabel Fakta:")
print(data_eks.info())
print("Statistik Deskriptif (Numerik):")
print(data_eks.describe())
print("Statistik Deskriptif Kategorikal:")
print(data_eks.describe(include=['object']))

#Cek Missing Value
miss_val = data_eks.isnull().sum()
miss_perc = (miss_val/len(data_eks)) * 100
print(pd.DataFrame({'Missing Values': miss_val, 'Percentage': miss_perc}))


# === Analisis Data Menggunakan Query SQL ===
cust_dim.createOrReplaceTempView("cust_dim")
product_dim.createOrReplaceTempView("product_dim")
loc_dim.createOrReplaceTempView("loc_dim")
time_dim.createOrReplaceTempView("time_dim")
fact_trans.createOrReplaceTempView("fact_trans")
print()
print("====================ANALISIS DATA MENGGUNAKAN QUERY SQL====================")

# 1. Produk terlaris berdasarkan kategori/sub-kategori
print("Produk Terlaris Berdasarkan Kategori")
query_1 = spark.sql("""
                    SELECT 
                        pd.Category,
                        pd.Product_Name,
                        SUM(ft.Sales) AS Total_Sales
                    FROM 
                        fact_trans ft
                    JOIN 
                        product_dim pd
                    ON 
                        ft.Product_ID = pd.Product_ID
                    GROUP BY 
                        pd.Category, pd.Product_Name
                    ORDER BY 
                        pd.Category ASC, Total_Sales DESC
                    """)
query_1.show()

print("Produk Terlaris Berdasarkan Sub-Kategori")
query_2 = spark.sql("""
                    SELECT 
                        pd.`Sub-Category`,
                        pd.Product_Name,
                        SUM(ft.Sales) AS Total_Sales
                    FROM 
                        fact_trans ft
                    JOIN 
                        product_dim pd
                    ON 
                        ft.Product_ID = pd.Product_ID
                    GROUP BY 
                        pd.`Sub-Category`, pd.Product_Name
                    ORDER BY 
                        pd.`Sub-Category` ASC, Total_Sales DESC
                    """)
query_2.show()

# 2. Segmetasi pelanggan berdasarkan lokasi dan total belanja
print("Segmentasi Pelanggan Berdasarkan Lokasi Geografis dan Total Belanja")
query_3 = spark.sql("""
                    SELECT 
                        l.Country,
                        l.Region,
                        l.State,
                        l.City,
                        c.Segment,
                        SUM(f.Sales) AS Total_Sales
                    FROM 
                        fact_trans f
                    JOIN 
                        cust_dim c ON f.Customer_ID = c.Customer_ID
                    JOIN 
                        loc_dim l ON f.Postal_Code = l.Postal_Code
                    GROUP BY 
                        l.Country, l.Region, l.State, l.City, c.Segment
                    ORDER BY 
                        Total_Sales DESC
                    """)
query_3.show()

# 3. Pola Penjualan
print("Pola Penjualan Bulanan:")
query_4 = spark.sql("""
                    SELECT 
                        t.Year,
                        t.Month,
                        SUM(f.Sales) AS Total_Sales
                    FROM 
                        fact_trans f
                    JOIN 
                        time_dim t ON f.Order_Date = t.Order_Date
                    GROUP BY 
                        t.Year, t.Month
                    ORDER BY 
                        t.Year, t.Month
                    """)
query_4.show()

print("Pola Penjualan Kuartalan:")
query_5 = spark.sql("""
                    SELECT 
                        t.Year,
                        t.Quarter,
                        SUM(f.Sales) AS Total_Sales
                    FROM 
                        fact_trans f
                    JOIN 
                        time_dim t ON f.Order_Date = t.Order_Date
                    GROUP BY 
                        t.Year, t.Quarter
                    ORDER BY 
                        t.Year, t.Quarter
                    """)
query_5.show()

print("Pola Penjualan Tahunan:")
query_6 = spark.sql("""
                    SELECT 
                        t.Year,
                        SUM(f.Sales) AS Total_Sales
                    FROM 
                        fact_trans f
                    JOIN 
                        time_dim t ON f.Order_Date = t.Order_Date
                    GROUP BY 
                        t.Year
                    ORDER BY 
                        t.Year
                    """)
query_6.show()

# 3. Perbandingan Penjualan
print("Perbandingan Penjualan Antar Lokasi")
query_7 = spark.sql("""
                    SELECT 
                        ld.Region AS Region,
                        ld.State AS State,
                        ld.City AS City,
                        SUM(ft.Sales) AS Total_Sales
                    FROM 
                        fact_trans ft
                    JOIN 
                        loc_dim ld
                    ON 
                        ft.Postal_Code = ld.Postal_Code
                    GROUP BY 
                        ld.Region, ld.State, ld.City
                    ORDER BY 
                        Total_Sales DESC
                    """)
query_7.show()