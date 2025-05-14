from spark_session import create_spark_session

def load_data():
    spark = create_spark_session()
    data_path = "data/tumor_classification_data.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    return df

if __name__ == "__main__":
    df = load_data()
    df.printSchema()
    df.show(5)
