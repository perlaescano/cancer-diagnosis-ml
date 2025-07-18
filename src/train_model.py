from preprocessing import preprocess_data
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from evaluation import evaluate_model

def split_data(df, train_ratio=0.7, seed=12321):
    """
    Split the preprocessed DataFrame into training and testing sets.
    """
    return df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)


def train_random_forest(train_df, test_df):
    """
    Train a Random Forest classifier and evaluate its performance.
    """
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=12321)
    model = rf.fit(train_df)
    predictions = model.transform(test_df)
    evaluate_model(predictions, model_name="RandomForestClassifier")
    return model

def train_logistic_regression(train_df, test_df):
    """
    Train a Logistic Regression model and evaluate its performance.
    """
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=100)
    model = lr.fit(train_df)
    predictions = model.transform(test_df)
    evaluate_model(predictions, model_name="LogisticRegression")
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

    # Train and evaluate models
    rf_model = train_random_forest(train_data, test_data)
    lr_model = train_logistic_regression(train_data, test_data)