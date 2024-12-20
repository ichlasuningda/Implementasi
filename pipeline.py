from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, to_date
from delta.tables import DeltaTable

#=== Inisialisasi SparkSession ===
spark = SparkSession.builder \
    .appName("Pipeline dengan Spark dan Delta Lake") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

spark.conf.set("spark.sql.debug.maxToStringFields", 1000)  # Ganti 1000 dengan nilai yang lebih besar jika diperlukan

#Cek Session
print("Spark Session Berhasil dibuat.")


#=== Pipeline Ingestion ===
#Path Data
hdfs_input = "hdfs://localhost:9000/dataMDIK/train.csv"

#Baca Data
print("Membaca data dari HDFS...")
df = spark.read.option("quote", "\"") \
        .option("escape", "\"") \
        .csv(hdfs_input, header=True, inferSchema=True)
print("Skema Data: ")
df.printSchema()

#Cek dan Hapus Duplikasi Data
total_data = df.count()
unique_data = df.dropDuplicates().count()
dup_data = total_data-unique_data
print("Data Duplikasi: ", dup_data)

#Hapus Duplikasi Data
if dup_data > 0:
    df_ing = df.dropDuplicates()
else:
    df_ing = df
    print("Tidak ada data yang dihapus")

#Ganti Nama Kolom
df_ing = df_ing.select([col(c).alias(c.replace(" ", "_")) for c in df_ing.columns])

print("Data sebelum transformasi: ")
df_ing.show(5)

#=== Pipeline ETL ===
#Transformasi Datatype
df_ing = df_ing.withColumn('Order_Date', to_date(col('Order_Date'), 'dd/MM/yyyy'))
df_trans = df_ing.withColumn('Ship_Date', to_date(col('Ship_Date'), 'dd/MM/yyyy'))
print("Skema Setelah Konversi: ")
df_trans.printSchema()

#Cari Missing Value
print("Informasi Missing Value Sebelum Ditangani:")
missing_values = df_trans.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df_trans.columns]
)
missing_values.show(truncate=False)


#Menangani Missing Value
data_med = df_ing.approxQuantile('Postal_Code', [0.5], 0)[0]
df_filled = df_ing.fillna({'Postal_Code': data_med})

print("Informasi Missing Value Setelah Ditangani dengan Nilai Median:")
missing_values = df_filled.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df_filled.columns]
)
missing_values.show(truncate=False)

#Cari Outlier
Q1, Q3 = df_filled.approxQuantile("Sales", [0.25, 0.75], 0.0)
lower_bound = Q1 - 1.5 * (Q3 - Q1)
upper_bound = Q3 + 1.5 * (Q3 - Q1)
outlier_count = df_filled.filter((col("Sales") < lower_bound) | (col("Sales") > upper_bound)).count()
print(f"Jumlah Outlier: {outlier_count}")

#Hapus Outlier
df_cleaned = df_filled.filter((col("Sales") >= lower_bound) & (col("Sales") <= upper_bound))
clean_count = (df_filled.count()) - outlier_count

print(f"Jumlah Data Setelah Penghapusan Outlier: {clean_count}")
print("Data Setelah Menghapus Outlier: ")
df_cleaned.show(5)

#=== Simpan Data Terstruktur Menggunakan Delta Lake ===
#Konfigurasi Path Delta Lake
delta_path = "hdfs://localhost:9000/dataMDIK/processed_data"

#Simpan Data Bersih
df_cleaned.write.format("parquet").mode("overwrite").save(delta_path)

print("Data terstruktur berhasil disimpan ke Delta Lake di:", delta_path)