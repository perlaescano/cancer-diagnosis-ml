from preprocessing import preprocess_data

def split_data(df, train_ratio=0.7, seed=42):
    """
    Split the preprocessed DataFrame into training and testing sets.
    """
    return df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

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
