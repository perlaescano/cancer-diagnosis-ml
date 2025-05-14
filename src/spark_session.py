from pyspark.sql import SparkSession

def create_spark_session(app_name="CancerDiagnosisML"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark
