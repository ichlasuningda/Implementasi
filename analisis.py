from pyspark.sql import SparkSession

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


#=== Koneksi MySQL ===
#Properti
mysql_url = "jdbc:mysql://localhost:3306/datamdik"
tabel = "data_retail"
conn_prop = {
    "user" : "root",
    "password" : "",
    "driver" : "com.mysql.cj.jdbc.Driver"
}

#Simpan Data ke MySQL
data.write.jdbc(url = mysql_url, table = tabel, mode = 'overwrite', properties = conn_prop)
print(f"Data Telah Berhasil Ditambahkan ke Tabel '{tabel}' di Data Warehouse MySQL.")

#Cek Data di MySQL
print("Cek Data di MySQL:")
df = spark.read.jdbc(url = mysql_url, table = tabel, properties = conn_prop)
df.show(5)