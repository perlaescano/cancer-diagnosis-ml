from load_data import load_data
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler

def preprocess_data():
    """
    Load data, clean, encode labels, and assemble features for Spark MLlib.
    Returns: Spark DataFrame with 'features' and 'label' columns.
    """
    # Load raw data using Spark
    df = load_data()

    # Drop 'id' column (not a feature) and remove any missing rows
    df_cleaned = df.drop("id").dropna()

    # Encode 'diagnosis': Malignant → 1, Benign → 0
    df_encoded = df_cleaned.withColumn("label", when(col("diagnosis") == "M", 1).otherwise(0)).drop("diagnosis")

    # Prepare the list of feature columns (exclude 'label')
    feature_cols = df_encoded.columns
    feature_cols.remove("label")

    # Assemble features into a single vector column (required by Spark MLlib)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_final = assembler.transform(df_encoded).select("features", "label")

    return df_final

if __name__ == "__main__":
    # Load, preprocess, and preview data
    df_preprocessed = preprocess_data()
    
    # Print schema and sample rows (no truncation for clarity)
    df_preprocessed.printSchema()
    df_preprocessed.show(5, truncate=False)
    
    # Show class distribution
    df_preprocessed.groupBy("label").count().show()
    
    # Show a few benign examples
    df_preprocessed.filter("label = 0").show(5, truncate=False)
