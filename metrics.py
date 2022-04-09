from sklearn import metrics
import torch
import torch.nn.functional as F
import numpy as np

def evaluate(y_test, y_pred_score):
    y_pred_score = torch.softmax(y_pred_score, dim = 1)
    _, y_pred_tags = torch.max(y_pred_score, dim = 1)
    y_pred = F.one_hot(y_pred_tags,num_classes = y_test.shape[1])
    y_test = y_test.cpu()
    y_pred = y_pred.cpu()
    accuracy = metrics.accuracy_score(y_test, y_pred)

    aucs = []
    auprs = []
    for cls in range(y_pred_score.shape[1]):
        aupr = metrics.average_precision_score(y_test[:,cls], y_pred_score[:,cls].detach().cpu())
        fpr, tpr, _ = metrics.roc_curve(y_test[:,cls], y_pred_score[:,cls].detach().cpu(), pos_label=1)
        aucs.append(metrics.auc(fpr, tpr))
        auprs.append(aupr)
    f1 = metrics.f1_score(y_test, y_pred,average='micro')
    precision = metrics.precision_score(y_test, y_pred,average='micro')
    recall = metrics.recall_score(y_test, y_pred,average='micro')
    
    score = {}
    score['accuracy'] = accuracy
    score['precision'] = precision
    score['auc'] = aucs
    score['mauc'] = np.mean(np.nan_to_num(aucs,nan=0.0))
    score['f1'] = f1
    score['aupr'] = auprs
    score['maupr'] = np.mean(np.nan_to_num(auprs,nan=0.0))
    score['recall'] = recall
    return score