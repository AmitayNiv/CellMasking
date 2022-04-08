import torch
import torch.nn as nn
import numpy as np
from models import Classifier,G_Model
import torch.optim as optim
from torch.utils.data import DataLoader

def train_classifier(args,device,data_obj,model,wandb_exp):
    
    train_losses = []
    val_losses =[]
    best_model_auc = 0


    if model == None:
        model = Classifier(len(data_obj.overlap ),dropout=args.dropout,number_of_classes=len(data_obj.labels.columns))
        model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=data_obj.class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.cls_lr)

    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=args.batch_size)
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=args.batch_size)



    for e in range(args.epochs):
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
################################################################################################################
                val_score = evaluate(y_validate_, y_pred_val, predictions)

    
                    
            train_losses.append(train_loss/len(train_batch))
            val_losses.append(val_loss)

            print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                  "Training Loss: {:.5f}.. ".format(train_loss/len(train_batch)),
                  "Val Loss: {:.5f}.. ".format(val_loss),
                  "Val AUC: {:.5f}.. ".format(val_score["auc"]))
            if val_score["auc"] >best_model_auc:
                best_model_auc = val_score["auc"]
                best_model_index = e+1
                # best_model_name = "/F_model_{:.3f}.pt".format(best_model_auc)
                # best_model_path = current_folder+best_model_name
                print("Model Saved, Auc ={:.4f} , Epoch ={}".format(best_model_auc,best_model_index))
                # torch.save(model,best_model_path)
                best_model = copy.deepcopy(model)
            # scheduler.step(val_score["auc"])

    best_model = best_model.cuda()#torch.load(best_model_path).cuda()
    with torch.no_grad():
      best_model.eval()
      y_pred_score = best_model(X_test)
      y_pred_score = y_pred_score.detach().cpu().numpy()
      y_pred_test = (y_pred_score> 0.5).astype(int)
      test_score = evaluate(y_test, y_pred_test, y_pred_score)
      print("Classifier test results:")
      print(test_score)
      return best_model

