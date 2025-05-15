from preprocessing import preprocess_data
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def split_data(df, train_ratio=0.7, seed=12321):
    """
    Split the preprocessed DataFrame into training and testing sets.
    """
    return df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

def train_random_forest(train_df, test_df):
    """
    Train and evaluate a Random Forest classifier.
    Prints Accuracy, Precision, Recall, and F1 Score.
    """
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=12321)
    model = rf.fit(train_df)

    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    print("=== Random Forest Evaluation ===")
    print(f"Accuracy:  {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"F1 Score:  {f1:.5f}")

    return model

if __name__ == "__main__":
    # Preprocess and get the final DataFrame
    df_preprocessed = preprocess_data()

    # Split into training and testing sets
    train_data, test_data = split_data(df_preprocessed)

    # Show summary
    print("Training set label distribution:")
    train_data.groupBy("label").count().show()

    print("Test set label distribution:")
    test_data.groupBy("label").count().show()

    print("Sample from training set:")
    train_data.show(5, truncate=False)

    print("Sample from test set:")
    test_data.show(5, truncate=False)

    # Train and evaluate the Random Forest classifier
    train_random_forest(train_data, test_data)