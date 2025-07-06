import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from collections import Counter

def weighted_clf_metrics(y_true, y_pred):
    """
    Calculate weighted classification metrics
    
    Parameters:
    y_true: true labels
    y_pred: predicted labels
    """
    # # Calculate weights based on class distribution
    # class_counts = Counter(y_true)
    # total_samples = len(y_true)
    # class_weights = {cls: count/total_samples for cls, count in class_counts.items()}
    
    # # Convert class weights to sample weights
    # sample_weights = np.array([class_weights[y] for y in y_true])
    # metrics = {
    #     "accuracy": "{:.2f}%".format(accuracy_score(y_true, y_pred, sample_weight=sample_weights) * 100),
    #     "precision": "{:.2f}%".format(precision_score(y_true, y_pred, average='weighted') * 100),
    #     "recall": "{:.2f}%".format(recall_score(y_true, y_pred, average='weighted') * 100),
    #     "f1": "{:.2f}%".format(f1_score(y_true, y_pred, average='weighted') * 100)
    # }
    output = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    metrics = {
        "accuracy": "{:.2f}%".format(output['accuracy'] * 100),
        "precision": "{:.2f}%".format(output['weighted avg']['precision'] * 100),
        "recall": "{:.2f}%".format(output['weighted avg']['recall'] * 100),
        "f1": "{:.2f}%".format(output['weighted avg']['f1-score'] * 100)
    }
    return metrics

def evaluate(filename):
    if '.csv' in filename:
        df = pd.read_csv(filename)
    elif '.json' in filename:
        df = pd.read_json(filename, orient='records')
    else:
        return "Format not supported in evaluation"
    
    y_pred = df['y_pred'].tolist()  # your predicted values
    y_true = df['y_true'].tolist()  # your true values
    print(classification_report(y_true=y_true, y_pred=y_pred, digits=4))
    output = weighted_clf_metrics(y_true=y_true, y_pred=y_pred)
    print("Eval on test:", output)
    

if __name__ == "__main__":
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("Please provide an argument")
        sys.exit(1)

    # Get the first argument
    filename = sys.argv[1]
    if os.path.exists(filename):
        evaluate(filename)



