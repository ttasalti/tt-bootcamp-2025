"""Split the raw JSONL shards by service type and clean each segment with PySpark.

Reads capstone.1.jsonl ... capstone.10.jsonl from $TT_DATA, writes one Parquet file per
service type under $TT_DATA and the cleaned versions under $TT_DATA/processed.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, median, mode, size, when

from config import DATA_DIR, SEGMENTS


def split_by_service_type(spark, shards):
    df = spark.read.json(shards)
    service_types = [row[0] for row in df.select("service_type").distinct().collect()]
    return {name: df.filter(col("service_type") == name) for name in service_types}


def clean(df, settings):
    df = df.drop(*settings["drop"])
    fills = {}
    for column in settings["fill_mode"]:
        fills[column] = df.select(mode(column)).collect()[0][0]
    for column in settings["fill_median"]:
        fills[column] = df.select(median(column)).collect()[0][0]
    df = df.fillna(fills).na.drop(subset=["tenure"])
    for column, dtype in df.dtypes:
        if dtype in ("bigint", "boolean"):
            df = df.withColumn(column, col(column).cast("int"))
    if "apps" in df.columns:
        df = df.withColumn("apps_count", size(col("apps"))).drop("apps")
    return df.drop("service_type")


def summarise(df, name):
    total = df.count()
    nulls = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    churn_rate = df.filter(col("churn") == 1).count() / total if total else 0.0
    print(f"{name}: {total} rows, churn rate {churn_rate:.4f}")
    for column, n in nulls.items():
        if n:
            print(f"  {column}: {n} nulls")


def main():
    spark = SparkSession.builder.appName("tt-churn-prepare").getOrCreate()
    shards = [f"{DATA_DIR}/capstone.{i}.jsonl" for i in range(1, 11)]
    for name, df in split_by_service_type(spark, shards).items():
        df.write.parquet(f"{DATA_DIR}/{name}.parquet", mode="overwrite")
    for segment, settings in SEGMENTS.items():
        raw = spark.read.parquet(f"{DATA_DIR}/{settings['file']}")
        processed = clean(raw, settings)
        summarise(processed, segment)
        processed.write.parquet(f"{DATA_DIR}/processed/{settings['file']}", mode="overwrite")


if __name__ == "__main__":
    main()
