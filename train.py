import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from models import Classifier,G_Model
from metrics import evaluate


def train_classifier(args,device,data_obj,model,wandb_exp):
    
    train_losses = []
    val_losses =[]
    best_model_auc = 0


    if model == None:
        model = Classifier(len(data_obj.colnames),dropout=args.dropout,number_of_classes=len(data_obj.labels.columns))
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()#weight=data_obj.class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.cls_lr)

    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=args.batch_size)
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=len(data_obj.val_dataset))
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))



    for e in range(args.cls_epochs):
        model.train()
        train_loss = 0
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()
            
            train_loss += train_loss.item()
        else:
            val_loss = 0
        
            with torch.no_grad():
                model.eval()
                val_epoch_loss = 0
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    y_val_pred = model(X_val_batch)     
                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_epoch_loss += val_loss.item()
                    val_score = evaluate(y_val_batch, y_val_pred)

    
                    
            # train_losses.append(train_loss/len(train_batch))
            # val_losses.append(val_loss)

            print("Epoch: {}/{}.. ".format(e+1, args.cls_epochs),
                  "Training Loss: {:.5f}.. ".format(train_loss/len(train_loader)),
                  "Val Loss: {:.5f}.. ".format(val_loss/len(val_loader)),
                  "Val mAUC: {:.5f}.. ".format(val_score["mauc"]),
                  "Val mAUPR: {:.5f}.. ".format(val_score["maupr"]),
                  "Val ACC: {:.5f}.. ".format(val_score["accuracy"]),
                  )
            if val_score["mauc"] >best_model_auc:
                best_model_auc = val_score["mauc"]
                best_model_index = e+1
                # best_model_name = "/F_model_{:.3f}.pt".format(best_model_auc)
                # best_model_path = current_folder+best_model_name
                print("Model Saved, Auc ={:.4f} , Epoch ={}".format(best_model_auc,best_model_index))
                # torch.save(model,best_model_path)
                best_model = copy.deepcopy(model)
            # scheduler.step(val_score["auc"])

    best_model = best_model.to(device)
    with torch.no_grad():
        best_model.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            y_pred_score = best_model(X_test_batch)
            test_score = evaluate(y_test_batch, y_pred_score)
            print("Classifier test results:")
            print(test_score)
    return best_model


def train_G(args,device,data_obj,classifier,model=None,wandb_exp=None):
    classifier.eval()


    train_losses = []
    val_losses_list =[]
    best_model_auc = 0

    if model == None:
        model = G_Model(len(data_obj.colnames))
        model = model.to(device)

    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=args.batch_size)
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=len(data_obj.val_dataset))
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))

    criterion = nn.CrossEntropyLoss()#weight=data_obj.class_weights.to(device))
    optimizer_G = optim.Adam(model.parameters(), lr=args.cls_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_G,max_lr=float(args.g_lr), steps_per_epoch=len(train_loader), epochs=args.g_epochs)



    for e in range(args.g_epochs):
        train_loss = 0
        loss_list = []
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer_G.zero_grad()
            mask = model(X_train_batch)
            cropped_features = X_train_batch * mask
            output = classifier(cropped_features)
            loss = criterion(output, y_train_batch) + mask.mean()
            loss.backward()
            optimizer_G.step()
            scheduler.step()
            train_loss += loss.item()
        else:
            val_losses = 0
        
            with torch.no_grad():
                model.eval()
                classifier.eval()
                val_epoch_loss = 0
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    mask_val = model(X_val_batch)   
                    print("mask mean = {}, max= {}, min = {}".format(mask_val.mean(),mask_val.max(),mask_val.min()))
                    cropped_features = X_val_batch * mask_val
                    output = classifier(cropped_features)
                    val_loss = criterion(output, y_val_batch) + mask_val.mean()
                    val_losses +=  val_loss.item()

                    val_score = evaluate(y_val_batch, output)
                print("Epoch: {}/{}.. ".format(e+1,  args.g_epochs),
                  "Training Loss: {:.5f}.. ".format(train_loss/len(train_loader)),
                  "Val Loss: {:.5f}.. ".format(val_losses/len(val_loader)),
                  "Val mAUC: {:.5f}.. ".format(val_score["mauc"]),
                  "Val mAUPR: {:.5f}.. ".format(val_score["maupr"]),
                  "Val ACC: {:.5f}.. ".format(val_score["accuracy"]))
                if val_score["mauc"] >best_model_auc:
                    best_model_auc = val_score["mauc"]
                    best_model_index = e+1
                    print("Model Saved,Epoch = {}".format(best_model_index))
                    best_G_model = copy.deepcopy(model)
                    # torch.save(model,best_model_path)	
    with torch.no_grad():
        best_G_model.eval()
        classifier.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            print("Results without G")
            y_pred_score = classifier(X_test_batch)
            test_score = evaluate(y_test_batch,y_pred_score)
            print(test_score)
            print("Results with G")
            mask_test = best_G_model(X_test_batch)
            cropped_features = X_test_batch * mask_test
            y_pred_score = classifier(cropped_features)
            test_score = evaluate(y_test_batch, y_pred_score)
            print(test_score)
    return best_G_model

