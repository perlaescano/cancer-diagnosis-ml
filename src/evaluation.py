from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# Custom short names for models
aliases = {
    "RandomForestClassifier": "rf",
    "LogisticRegression": "lr"
}

def evaluate_model(predictions, model_name="Model"):
    """
    Evaluate predictions and print/save results to the  /results directory.
    """
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    result_text = (
        f"=== {model_name} Evaluation ===\n"
        f"Accuracy:  {accuracy:.5f}\n"
        f"Precision: {precision:.5f}\n"
        f"Recall:    {recall:.5f}\n"
        f"F1 Score:  {f1:.5f}\n"
    )

    print(result_text)

    # Resolve project-level /results folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Extract simple model name from full name
    if "(" in model_name:
        base_name = model_name.split("(")[0].strip()
    else:
        base_name = model_name.strip()

    suffix = "cross_validation" if "CV" in model_name else "evaluation"
    model_abbr = aliases.get(base_name, base_name.lower())
    filename = f"{model_abbr}_{suffix}.txt"

    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(result_text)
