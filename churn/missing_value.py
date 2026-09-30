import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, size, mode, median

DATA_DIR = os.environ.get("TT_DATA", "data")

def read_parquet_and_count_nulls(parquet_paths):
    """
    Read each Parquet file and report its row count and the number of nulls per column.
    """
    spark = SparkSession.builder.appName("Parquet_Statistics").getOrCreate()
    
    parquet_data = {}
    
    for path in parquet_paths:
        df = spark.read.parquet(path)
        total_rows = df.count()

        
        # Null count per column
        na_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
        na_counts_dict = na_counts.collect()[0].asDict()
        
        print(f"Parquet file: {path}")
        print(f"Rows: {total_rows}")
        print("Nulls per column:")
        for column, na_count in na_counts_dict.items():
            print(f"  {column}: {na_count}")
        print("\n")
        
        parquet_data[path] = df
    
    return parquet_data

def drop_unwanted_columns(parquet_data):
    """
    Drop the columns that do not apply to a segment (call metrics for broadband,
    payment columns for prepaid, top-up count for postpaid).
    """
    for path, df in parquet_data.items():
        if "Broadband.parquet" in path:
            parquet_data[path] = df.drop("avg_call_duration", "call_drops", "roaming_usage")
        elif "Prepaid.parquet" in path:
            parquet_data[path] = df.drop("auto_payment","overdue_payments")
        elif "Postpaid.parquet" in path:
            parquet_data[path] = df.drop("avg_top_up_count")
            
    return parquet_data

def fill_na_values(parquet_data):
    """
    Fill missing values per segment: auto_payment with the mode, numeric columns with the
    median, and drop rows with a missing tenure.
    """
    for path, df in parquet_data.items():
        if "Broadband.parquet" in path:
            mode_payment = df.select(mode("auto_payment")).collect()[0][0]
            median_charge = df.select(median("monthly_charge")).collect()[0][0]
            median_data = df.select(median("data_usage")).collect()[0][0]
            df = df.fillna({"auto_payment": mode_payment, "monthly_charge": median_charge,
                            "data_usage": median_data})
            df = df.na.drop(subset=["tenure"])
        elif "Postpaid.parquet" in path:
            mode_payment = df.select(mode("auto_payment")).collect()[0][0]
            median_call = df.select(median("avg_call_duration")).collect()[0][0]
            median_data = df.select(median("data_usage")).collect()[0][0]
            median_charge = df.select(median("monthly_charge")).collect()[0][0]
            df = df.fillna({"auto_payment": mode_payment,"avg_call_duration": median_call,
                            "data_usage": median_data, "monthly_charge": median_charge})
            df = df.na.drop(subset=["tenure"])
        elif "Prepaid.parquet" in path:
            median_call = df.select(median("avg_call_duration")).collect()[0][0]
            median_data = df.select(median("data_usage")).collect()[0][0]
            median_charge = df.select(median("monthly_charge")).collect()[0][0]
            df = df.fillna({"avg_call_duration": median_call, "data_usage": median_data, 
                            "monthly_charge": median_charge})
            df = df.na.drop(subset=["tenure"])
        
        parquet_data[path] = df
    
    return parquet_data

def preprocess_columns(parquet_data):
    """
    Replace the 'apps' list with its length, drop 'service_type', and cast bigint and
    boolean columns to int.
    """
    for path, df in parquet_data.items():
        for column, dtype in df.dtypes:
            if dtype == "bigint":
                df = df.withColumn(column, col(column).cast("int"))
            elif dtype == "boolean":
                df = df.withColumn(column, col(column).cast("int"))
        
        if "apps" in df.columns:
            df = df.withColumn("apps_count", size(col("apps"))).drop("apps")
        if "service_type" in df.columns:
            df = df.drop("service_type")
        parquet_data[path] = df
    
    return parquet_data

def write_parquet(parquet_data, output_dir=None):
    """
    Write the processed Parquet files.
    """
    output_dir = output_dir or f"{DATA_DIR}/processed"
    for path, df in parquet_data.items():
        filename = path.split("/")[-1]
        df.write.parquet(f"{output_dir}/{filename}", mode="overwrite")
        print(f"Processed Parquet saved: {output_dir}/{filename}")

if __name__ == "__main__":
    parquet_files = [f"{DATA_DIR}/Broadband.parquet", f"{DATA_DIR}/Postpaid.parquet", f"{DATA_DIR}/Prepaid.parquet"]
    parquet_data = read_parquet_and_count_nulls(parquet_files)
    parquet_data = drop_unwanted_columns(parquet_data)
    parquet_data = fill_na_values(parquet_data)
    parquet_data = preprocess_columns(parquet_data)
    write_parquet(parquet_data)
