# Cancer Diagnosis Using Machine Learning

## Objective
Develop and evaluate machine learning models to predict whether a tumor is benign (B) or malignant (M) using a dataset of clinical variables.

## Dataset Details
- **Samples**: 569
- **Features**: 20 clinical variables
- **Labels**: B (Benign), M (Malignant)
- **Source**: Provided on Brightspace

## Requirements
- Python >=3.7
- pyspark>=3.0.0
- pandas>=1.0.0
- matplotlib>=3.0.0
- seaborn>=0.10.0


## Installation

It is recommended to use Python 3.9+.

Install the required dependencies with pip:

```bash
pip install pyspark pandas matplotlib seaborn
```

## Project Structure
```md
├── data
│   └── tumor_classification_data.csv
├── notebooks
│   └── eda.ipynb
├── REPORT.md
├── README.md
├── results
│   ├── lr_cross_validation.txt
│   ├── lr_evaluation.txt
│   ├── rf_cross_validation.txt
│   └── rf_evaluation.txt
└── src
    ├── __init__.py
    ├── cross_validation.py
    ├── evaluation.py
    ├── load_data.py
    ├── preprocessing.py
    ├── spark_session.py
    └── train_model.py
```

## Development Setup

To ensure VS Code recognizes imports from the `src/` folder as the root of your Python package, create a `.vscode` directory in the project root and add a `settings.json` file:

```bash
mkdir -p .vscode
touch .vscode/settings.json
```
Then add the following to `.vscode/settings.json`

```json
{
  "python.analysis.extraPaths": ["src"]
}
```
## How to Run

Follow the steps below to execute the different parts of the project:
1. **Start a Spark session**  
   This is handled internally by the other scripts using `src/spark_session.py`. You do not need to run this directly.

2. **Load and preview the data**  
To verify the dataset is loading correctly, run:

```bash
  python src/load_data.py
```
3. **Preprocess the data**
This script handles label encoding and feature vector assembly:
```bash
  python src/preprocessing.py

```
4. **Train models and evaluate**
This trains Random Forest and Logistic Regression models and saves evaluation results to the `results/ folder:`

```bash
  python src/train_model.py
```
5. **Run Cross-validation**
Applies 3-fold cross-validation to both models and saves the corresponding metrics:
```bash
  python src/cross_validation.py
```

## Evaluation Metrics 
  - F1 Score
  - Precision
  - Recall
  - Accuracy
  - Cross-validation



