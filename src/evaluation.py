from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

def evaluate_model(predictions, model_name="Model"):
    """
    Evaluate predictions and print/save results to the project-level /results folder.
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

    # Get absolute path to project-level "results" folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{model_name.replace(' ', '_').lower()}_evaluation.txt"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(result_text)
