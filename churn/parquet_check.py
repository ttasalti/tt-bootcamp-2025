import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

DATA_DIR = os.environ.get("TT_DATA", "data")

def print_parquet_column_info(parquet_paths):
    """
    Print column types, null counts and the churn rate of each processed Parquet file.
    """
    spark = SparkSession.builder.appName("Parquet_Column_Info").getOrCreate()
    
    for path in parquet_paths:
        df = spark.read.parquet(f"{DATA_DIR}/processed/{path.split('/')[-1]}")
        
        print(f"Parquet file: {path}")
        print("Column | Type | Nulls")

        print("--------------------------------------------------")
        
        na_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        for column, na_count in na_counts.items():
            dtype = dict(df.dtypes)[column]
            print(f"{column} | {dtype} | {na_count}")
        
        # Share of churned customers
        total_count = df.count()
        churn_count = df.filter(col("churn") == 1).count()
        churn_ratio = (churn_count / total_count) if total_count > 0 else 0
        print(f"Churn rate: {churn_ratio:.4f}")
        print("\n")

if __name__ == "__main__":
    parquet_files = ["Broadband.parquet", "Postpaid.parquet", "Prepaid.parquet"]
    print_parquet_column_info(parquet_files)
