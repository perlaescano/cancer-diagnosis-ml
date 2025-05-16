from preprocessing import preprocess_data
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from evaluation import evaluate_model

def cross_validate_model(model, df, param_grid, num_folds=3):
    """
    Perform k-fold cross-validation and evaluate the best model.
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    cv = CrossValidator(
        estimator=model,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=12321
    )

    cv_model = cv.fit(df)

    predictions = cv_model.transform(df)
    evaluate_model(predictions, model_name=f"{model.__class__.__name__} (CV)")

    return cv_model

if __name__ == "__main__":
    df_preprocessed = preprocess_data()

    # Cross-validation for Random Forest
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=12321)
    rf_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    print("\n=== Random Forest Cross-Validation ===")
    best_rf_model = cross_validate_model(rf, df_preprocessed, rf_grid)

    # Cross-validation for Logistic Regression
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=100)
    lr_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.0, 0.1]) \
        .build()

    print("\n=== Logistic Regression Cross-Validation ===")
    best_lr_model = cross_validate_model(lr, df_preprocessed, lr_grid)
