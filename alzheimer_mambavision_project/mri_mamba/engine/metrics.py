from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return accuracy, macro_f1
