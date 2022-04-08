from sklearn import metrics

def evaluate(y_test, y_pred, y_pred_score=None):
    y_test = y_test.cpu()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(y_pred_score)
    if y_pred_score is None:
        y_pred_score = y_pred
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    print(fpr, tpr, thresholds)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    aupr = metrics.average_precision_score(y_test, y_pred_score)
    score = {}
    score['accuracy'] = accuracy
    score['precision'] = precision
    score['auc'] = auc
    score['f1'] = f1
    score['aupr'] = aupr
    score['recall'] = recall
    return score