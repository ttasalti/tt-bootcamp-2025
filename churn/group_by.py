import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

DATA_DIR = os.environ.get("TT_DATA", "data")

def group_by_service_type(file_paths):
    """
    Read the JSON files and split them by service_type.
    """
    spark = SparkSession.builder.appName("JSON_to_CSV_with_Lists").getOrCreate()
    
    # Read all files into one dataframe
    df = spark.read.json(file_paths)
    
    # One dataframe per service type
    service_types = df.select("service_type").distinct().rdd.flatMap(lambda x: x).collect()

    grouped_data = {service_type: df.filter(col("service_type") == service_type) for service_type in service_types}
    
    return grouped_data

def save_to_parquet(grouped_data, output_path):
    """
    Write one Parquet file per service type.
    """
    for service_type, df in grouped_data.items():
        df.write.parquet(f"{output_path}/{service_type}.parquet", mode="overwrite")

if __name__ == "__main__":
    file_paths = [f"{DATA_DIR}/capstone.{i}.jsonl" for i in range(1, 11)]
    parquet_output_path = DATA_DIR
    grouped_data = group_by_service_type(file_paths)
    save_to_parquet(grouped_data, parquet_output_path)
