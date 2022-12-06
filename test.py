import torch
from torch.utils.data import DataLoader
import numpy as np

from metrics import evaluate

def test(classifier,g_model,device,data_obj,test_H=False,model_name=""):
    """
    test models on test data

    Arguments:
    classifier [obj] - trained F/H model
    g_model [obj] - trained G model
    data_obj [obj] - Data object
    device


    """
    test_loader = DataLoader(dataset=data_obj.all_dataset, batch_size=len(data_obj.all_dataset))
    print(f"Test results on {data_obj.data_name} dataset")
    with torch.no_grad():
        g_model.eval()
        classifier.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            mask_test = g_model(X_test_batch)
            cropped_features = X_test_batch * mask_test
            if test_H:
                X_test_batch_bin = torch.where(X_test_batch==0, 1, 0)
                cropped_features_neg = X_test_batch_bin *mask_test
                cropped_features = torch.concat((cropped_features,cropped_features_neg),dim=1)

            print(f"{model_name} WITHOUT G Results")
            if test_H:
                X_test_batch_bin = torch.where(X_test_batch==0, 1, 0)
                X_test_batch = torch.concat((X_test_batch,X_test_batch_bin),dim=1)
            y_pred_score = classifier(X_test_batch)
            test_score = evaluate(y_test_batch, y_pred_score)
            print(test_score)

            print(f"{model_name} Results")
            y_pred_score = classifier(cropped_features)
            test_score = evaluate(y_test_batch, y_pred_score)

            print(test_score)
            res_dict = {
            f"{model_name} mAUC":test_score["mauc"],f"{model_name} wegAUC":test_score["weight_auc"],f"{model_name} medAUC":test_score["med_auc"],
            f"{model_name} mAUPR":test_score["maupr"],f"{model_name} wegAUPR":test_score["weight_aupr"],
            f"{model_name} medAUPR":test_score["med_aupr"],f"{model_name} Accuracy":test_score["accuracy"],f"{model_name} mAccuracy":test_score["maccuracy"],f"{model_name} wegAccuracy":test_score["weight_accuracy"]
            }
    return res_dict
    

def test_xgb(xgb_cls,data_obj,device):
    """
    test XGB model on test data

    Arguments:
    xgb_cls [obj] - trained XGB model
    data_obj [obj] - Data object
    device

    """
    print(f"XGB Test Results on {data_obj.data_name}")
    X_test = np.array(data_obj.all_dataset.X_data)
    y_test = np.array(data_obj.all_dataset.y_data)
    y_pred_score =  xgb_cls.predict_proba(X_test)
    y_pred_score = torch.from_numpy(y_pred_score).to(device)
    test_score = evaluate(y_test,y_pred_score)
    print(test_score)
    res_dict = {"XGB mAUC":test_score["mauc"],"XGB wegAUC":test_score["weight_auc"],"XGB medAUC":test_score["med_auc"],
            "XGB mAUPR":test_score["maupr"],"XGB wegAUPR":test_score["weight_aupr"],"XGB medAUPR":test_score["med_aupr"],
            "XGB Accuracy":test_score["accuracy"],"XGB mAccuracy":test_score["maccuracy"],"XGB wegAccuracy":test_score["weight_accuracy"]}
    return res_dict


def test_rf(rf_cls,data_obj,device):
    """
    test XGB model on test data

    Arguments:
    rf_cls [obj] - trained RF model
    data_obj [obj] - Data object
    device

    """
    print(f"Random Forest Test Results on {data_obj.data_name}")
    X_test = np.array(data_obj.all_dataset.X_data)
    y_test = np.array(data_obj.all_dataset.y_data)
    y_pred_score =  rf_cls.predict_proba(X_test)
    y_pred_score = torch.from_numpy(y_pred_score).to(device)
    test_score = evaluate(y_test,y_pred_score)
    print(test_score)
    res_dict = {"RF mAUC":test_score["mauc"],"RF wegAUC":test_score["weight_auc"],"RF medAUC":test_score["med_auc"],
            "RF mAUPR":test_score["maupr"],"RF wegAUPR":test_score["weight_aupr"],"RF medAUPR":test_score["med_aupr"],
            "RF Accuracy":test_score["accuracy"],"RF mAccuracy":test_score["maccuracy"],"RF wegAccuracy":test_score["weight_accuracy"]}
    return res_dict

