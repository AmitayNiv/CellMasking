from statistics import mean
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import scipy.stats # for creating a simple dataset 
from torch.utils.data import Dataset,DataLoader

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.datasets import make_moons
import copy
import pandas as pd
import random
from scipy.stats import norm

from metrics import evaluate


class Classifier(nn.Module):
    def __init__(self, D_in,dropout= 0,number_of_classes= 1,first_division = 2):
        super(Classifier, self).__init__()
        
        # self.fc1 = nn.Linear(D_in, (D_in//first_division))
        # self.fc2 = nn.Linear((D_in//first_division),(D_in//(first_division)))
        # self.fc3 = nn.Linear((D_in//(first_division)), (D_in//(first_division*2)))
        # self.fc4 = nn.Linear((D_in//(first_division*2)), (D_in//(first_division*2)))
        # self.fc5= nn.Linear((D_in//(first_division*2)), number_of_classes)
        self.fc1 = nn.Linear(D_in, (D_in*100))
        self.fc2 = nn.Linear((D_in*100),(D_in*10))
        self.fc3 = nn.Linear((D_in*10), (D_in*2))
        # self.fc4 = nn.Linear((D_in*10), (D_in*2))
        self.fc5= nn.Linear((D_in*2), number_of_classes)
        self.drop = nn.Dropout(p=dropout)
        self.selu = nn.Tanh()
             
    def forward(self, x):
        x = self.selu(self.fc1(x))
        x = self.drop(x)
        x = self.selu(self.fc2(x))
        x = self.drop(x)
        x = self.selu(self.fc3(x))
        # x = self.drop(x)
        # x = self.selu(self.fc4(x))
        x = self.fc5(x)

        return x

class G_Model(nn.Module):
    def __init__(self, input_dim,first_division = 2):
        super(G_Model, self).__init__()
        # Encoder: affine function
        # self.fc1 = nn.Linear(input_dim,input_dim)
        # self.fc2 = nn.Linear(input_dim, input_dim//(first_division))
        # self.fc3 = nn.Linear(input_dim//(first_division), input_dim//(first_division*2))
        # # Decoder: affine function
        # self.fc4 = nn.Linear( input_dim//(first_division*2),input_dim//(first_division))
        # self.fc5 = nn.Linear( input_dim//(first_division),input_dim//(first_division))
        # self.fc6 = nn.Linear(input_dim//first_division, input_dim)
        self.fc1 = nn.Linear(input_dim,input_dim*100)
        self.fc2 = nn.Linear(input_dim*100, input_dim*10)
        self.fc3 = nn.Linear(input_dim*10, input_dim*2)
        # Decoder: affine function
        self.fc4 = nn.Linear( input_dim*2,input_dim*10)
        self.fc5 = nn.Linear( input_dim*10,input_dim*100)
        self.fc6 = nn.Linear(input_dim*100, input_dim)
        self.sig = nn.Sigmoid()
        self.selu = nn.ReLU()

    def forward(self, a):
        x = self.selu(self.fc1(a))
        x = self.selu(self.fc2(x))
        x = self.selu(self.fc3(x))
        x = self.selu(self.fc4(x))
        x = self.selu(self.fc5(x))
        logits = self.fc6(x)
        mask = self.sig(logits)
        return mask


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def train_classifier(device,data_obj,model):
    """
    Train F classifier

    Arguments:
    args [obj] - Arguments
    device
    data_obj [obj] - Data object
    model [obj] - pretraind/initialized model to train
    wandn_exp - None/ weight and baises object

    Return:
    best_model [obj] - trained F model
    res_dict [dict] -  model results on test set
    """
    print("\nStart training classifier")
    train_losses = []
    val_losses =[]
    best_model_auc = 0
    lr = 0.01
    epochs =50

    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=len(data_obj.train_dataset))
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=len(data_obj.val_dataset))
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))



    criterion = nn.MSELoss()#nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)


    global_step = 0
    for e in range(epochs):
        model.train()
        train_loss = 0
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            

            y_train_pred = model(X_train_batch)
            loss = criterion(y_train_pred, y_train_batch)
            loss.backward()

            

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            
            train_loss += loss.item()

            global_step += 1
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
                    # val_score = evaluate(y_val_batch, y_val_pred)
                    val_r2 = r2_score(y_val_batch.detach().cpu(),y_val_pred.detach().cpu())
                    val_mse = mean_squared_error(y_val_batch.detach().cpu(),y_val_pred.detach().cpu())

    

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.5f}.. ".format(train_loss/len(train_loader)),
                "Val MSE: {:.5f}.. ".format(val_mse),
                "Val R2: {:.5f}.. ".format(val_r2))
                # "Val mAUPR: {:.5f}.. ".format(val_score["maupr"]),
                # "Val wegAUPR: {:.5f}.. ".format(val_score["weight_aupr"]),
                # "Val medAUPR: {:.5f}.. ".format(val_score["med_aupr"]),
                # "Val ACC: {:.5f}.. ".format(val_score["accuracy"])
            

            if val_r2>best_model_auc:
                best_model_auc = val_r2
                # best_model_res = val_score.copy()
                best_model_index = e+1
                print("Model Saved, Auc ={:.4f} , Epoch ={}".format(best_model_auc,best_model_index))
                best_model = copy.deepcopy(model)
    best_model = best_model.to(device)
    with torch.no_grad():
        best_model.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            y_pred_score = best_model(X_test_batch)
            test_r2 = r2_score(y_test_batch.detach().cpu(),y_pred_score.detach().cpu())
            test_mse = mean_squared_error(y_test_batch.detach().cpu(),y_pred_score.detach().cpu())
            # test_score = evaluate(y_test_batch, y_pred_score)
            # print("Classifier test results:")
            # print(test_score)
            print("Val MSE: {:.5f}.. ".format(test_mse),
                "Val R2: {:.5f}.. ".format(test_r2))
    return best_model


def train_G(device,data_obj,classifier,model=None):
    """
    Train G model

    Arguments:
    args [obj] - Arguments
    device
    data_obj [obj] - Data object
    classifier [obj] - pretraind F model 
    wandn_exp - None/ weight and baises object

    Return:
    best_G_model [obj] - trained G model
    res_dict [dict] -  model results on test set
    """
    print("\nStart training G model")
    classifier.eval()

    train_losses = []
    val_losses_list =[]
    best_model_auc = 0
    cls_lr = 0.02
    g_lr = 0.02
    epochs = 100
    batch_factor = 1
    weight_decay=5e-4

    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=60)
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=len(data_obj.val_dataset))
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))



    criterion = nn.MSELoss()
    # optimizer = optim.Adam(classifier.parameters(), lr=cls_lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=float(cls_lr), steps_per_epoch=len(train_loader), epochs=epochs)
    optimizer_G = optim.Adam(model.parameters(), lr=g_lr,weight_decay=weight_decay)
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G,max_lr=float(g_lr),steps_per_epoch=len(train_loader), epochs=epochs)
    # scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,0.5)


    global_step = 0
    for e in range(epochs):
        classifier.eval()
        train_loss = 0
        loss_list = []
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            mask = model(X_train_batch)
            cropped_features = X_train_batch * mask
            output = classifier(cropped_features)
            loss = criterion(output, y_train_batch) + mask.mean()
            loss.backward()
            
            # if (global_step + 1) % batch_factor == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     scheduler.step()

        # if (global_step + 1) % (batch_factor) == 0:
            optimizer_G.step()
            optimizer_G.zero_grad()
            scheduler_G.step()
            
            train_loss += loss.item()
            global_step +=1
        else:
            val_losses = 0
        
            with torch.no_grad():
                model.eval()
                classifier.eval()
                val_epoch_loss = 0
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    mask_val = model(X_val_batch)   
                    print("mask mean = {:.5f}, max= {:.5f}, min = {:.5f}".format(mask_val.mean(),mask_val.max(),mask_val.min()))
                    cropped_features = X_val_batch * mask_val
                    output = classifier(cropped_features)
                    val_loss = criterion(output, y_val_batch) + mask_val.mean()
                    val_losses +=  val_loss.item()
                    # val_score = evaluate(y_val_batch, output)
                    val_r2 = r2_score(y_val_batch.detach().cpu(),output.detach().cpu())
                    val_mse = mean_squared_error(y_val_batch.detach().cpu(),output.detach().cpu())

    

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.5f}.. ".format(train_loss/len(train_loader)),
                "Val MSE: {:.5f}.. ".format(val_mse),
                "Val R2: {:.5f}.. ".format(val_r2))


            
            if val_r2 >=best_model_auc:
                best_model_auc = val_r2
                # best_model_res = val_score.copy()
                best_model_index = e+1
                print("Model Saved, Auc ={:.4f} , Epoch ={}".format(best_model_auc,best_model_index))
                best_G_model = copy.deepcopy(model)


    with torch.no_grad():
        best_G_model.eval()
        classifier.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            print("Results without G")
            y_pred_score = classifier(X_test_batch)
            test_r2 = r2_score(y_test_batch.detach().cpu(),y_pred_score.detach().cpu())
            test_mse = mean_squared_error(y_test_batch.detach().cpu(),y_pred_score.detach().cpu())
            # test_score = evaluate(y_test_batch, y_pred_score)
            # print("Classifier test results:")
            # print(test_score)
            print("Val MSE: {:.5f}.. ".format(test_mse),
                "Val R2: {:.5f}.. ".format(test_r2))
            print("Results with G")
            mask_test = best_G_model(X_test_batch)
            cropped_features = X_test_batch * mask_test
            y_pred_score = classifier(cropped_features)
            test_r2 = r2_score(y_test_batch.detach().cpu(),y_pred_score.detach().cpu())
            test_mse = mean_squared_error(y_test_batch.detach().cpu(),y_pred_score.detach().cpu())
            # test_score = evaluate(y_test_batch, y_pred_score)
            # print("Classifier test results:")
            # print(test_score)
            print("Val MSE: {:.5f}.. ".format(test_mse),
                "Val R2: {:.5f}.. ".format(test_r2))

    return best_G_model


class sim_Data():
    def __init__(self,n_size=120,p_size =20) -> None:
        X_data, y_data=  self.create_eq1_data(n_size,p_size)
        # # X_train, X_test, y_train, y_test = X_data[:n_size//4],X_data[n_size//2:],y_data[:n_size//2],y_data[n_size//2:]#train_test_split(X_data, y_data, train_size=0.5)


        # X_train = np.concatenate([X_data[:n_size//4], X_data[n_size//2:n_size//2+n_size//4]], axis=0)
        # y_train = np.concatenate([y_data[:n_size//4], y_data[n_size//2:n_size//2+n_size//4]], axis=0)

        # X_test_f = np.concatenate([X_data[n_size//4:n_size//2], X_data[n_size//2+n_size//4:]], axis=0)
        # y_test_f = np.concatenate([y_data[n_size//4:n_size//2], y_data[n_size//2+n_size//4:]], axis=0)

        # X_test = X_test_f[::2]
        # y_test = y_test_f[::2]

        # X_valid = X_test_f[1::2]
        # y_valid = y_test_f[1::2]

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.5,random_state=42)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, train_size=0.5,random_state=42)

        self.train_dataset  = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        self.val_dataset   = ClassifierDataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).float())
        self.test_dataset   = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        self.all_dataset   = ClassifierDataset(torch.from_numpy(X_data).float(), torch.from_numpy(y_data).float())
        print(X_data.shape)
        print(y_data.shape)

        
    def create_eq1_data(self,n,p):
        x1=(np.random.uniform(0,1,n)).reshape(-1,1)
        x2=(np.random.uniform(0,1,n)).reshape(-1,1)
        x3=(np.random.uniform(0,1,n)).reshape(-1,1)
        x4=(np.random.uniform(0,1,n)).reshape(-1,1)
        x5=(np.random.uniform(0,1,n)).reshape(-1,1)

        y1 = (-2*(x1[:n//2])+(x2[:n//2])-0.5*x3[:n//2])
        y2 = -0.5*x3[n//2:]+4*x4[n//2:]-2*x5[n//2:]
        relevant = np.hstack((x1,x2,x3,x4,x5))
        noise_vector = scipy.stats.norm.rvs(loc=0, scale=1, size=[n,p-5])
        data = np.concatenate([relevant, noise_vector], axis=1)
        y = np.concatenate((y1, y2), axis=0)
        return data, y.astype(np.float32)
    
    def create_sin_dataset(self,n,p):
        x1=5*(np.random.uniform(0,1,n)).reshape(-1,1)
        x2=5*(np.random.uniform(0,1,n)).reshape(-1,1)
        y=np.sin(x1)*np.cos(x2)**3
        relevant=np.hstack((x1,x2))
        noise_vector = scipy.stats.norm.rvs(loc=0, scale=1, size=[n,p-2])
        data = np.concatenate([relevant, noise_vector], axis=1)
        return data, y.astype(np.float32)

    def create_twomoon_dataset(self,n, p):
        relevant, y = make_moons(n_samples=n, shuffle=True, noise=0.1, random_state=None)
        print(y.shape)
        y = np.concatenate([np.logical_not(y).astype(int).reshape(-1,1),y.reshape(-1,1)],axis=1)
        noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
        data = np.concatenate([relevant, noise_vector], axis=1)
        print(data.shape)
        return data, y


def run():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(f'Using device {device}')
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    n_size = 1000
    p_size = 10#5
    data_obj = sim_Data(n_size=n_size,p_size=p_size)


    cls = Classifier(p_size ,dropout=0.2,number_of_classes=1,first_division=1)
    cls = cls.to(device)
    print("Initializing G model")
    g_model = G_Model(p_size,first_division=2)
    g_model = g_model.to(device)

    cls = train_classifier(device=device,data_obj=data_obj,model=cls)
    
    g_model= train_G(device,data_obj=data_obj,classifier=cls,model=g_model)
    print()


    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset),shuffle=False)
    first_batch = True
    with torch.no_grad():
        g_model.eval()
        for X_batch, y_batch in dataset_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = g_model(X_batch)
            mask.requires_grad = False
            if first_batch:
                mask_arr = mask
                first_batch = False
            else:
                mask_arr = torch.cat((mask_arr,mask), 0)
        
        mask_arr = torch.cat((mask_arr,y_batch), 1)
    mask = np.array(mask_arr.detach().cpu())
    df = pd.DataFrame(mask)
    print()

    # import matplotlib.pyplot as plt 
    # f,ax = plt.subplots(1,3,figsize=(10,5))
    # dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset),shuffle=False)
    # with torch.no_grad():
    #     for X_batch, y_batch in dataset_loader:
    #         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #         output = cls(X_batch)
    #         y_pred_score = torch.softmax(output, dim = 1)
    #         y_pred_tags = torch.max(y_pred_score, dim = 1)

    #         mask_test = g_model(X_batch)
    #         cropped_features = X_batch * mask_test
    #         y_pred_score = cls(cropped_features)
    #         y_pred_score = torch.softmax(y_pred_score, dim = 1)
    #         y_pred_tags_g = torch.max(y_pred_score, dim = 1)

    #         ax[0].scatter(x=X_batch[:,0].detach().cpu(), y=X_batch[:,1].detach().cpu(), s=150, c=y_batch[:,1].detach().cpu().reshape(-1),alpha=0.4,cmap=plt.cm.get_cmap('RdYlBu'),)
    #         ax[0].set_xlabel('$x_1$',fontsize=20)
    #         ax[0].set_ylabel('$x_2$',fontsize=20)
    #         ax[0].set_title('Target y')
    #         ax[1].scatter(x=X_batch[:,0].detach().cpu(), y=X_batch[:,1].detach().cpu(), s=150, c=y_pred_tags.indices.detach().cpu().reshape(-1),alpha=0.4,cmap=plt.cm.get_cmap('RdYlBu'),)
    #         ax[1].set_xlabel('$x_1$',fontsize=20)
    #         ax[1].set_ylabel('$x_2$',fontsize=20)
    #         ax[1].set_title('Classification output ')
    #         ax[2].scatter(x=X_batch[:,0].detach().cpu(), y=X_batch[:,1].detach().cpu(), s=150, c=y_pred_tags_g.indices.detach().cpu().reshape(-1),alpha=0.4,cmap=plt.cm.get_cmap('RdYlBu'),)
    #         ax[2].set_xlabel('$x_1$',fontsize=20)
    #         ax[2].set_ylabel('$x_2$',fontsize=20)
    #         ax[2].set_title('G Classification output ')
    #         plt.tick_params(labelsize=10)
    #         plt.savefig("fig.png")
    #         print()


if __name__ == '__main__':
    run()