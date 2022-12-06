import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, D_in,dropout= 0,number_of_classes= 1,first_division = 2):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(D_in, (D_in//first_division))
        self.fc2 = nn.Linear((D_in//first_division),(D_in//(first_division*2)))
        self.fc3 = nn.Linear((D_in//(first_division*2)), (D_in//(first_division*4)))
        self.fc4= nn.Linear((D_in//(first_division*4)), number_of_classes)
        self.drop = nn.Dropout(p=dropout)
        self.selu = nn.SELU()
             
    def forward(self, x):
        x = self.selu(self.fc1(x))
        x = self.drop(x)
        x = self.selu(self.fc2(x))
        x = self.selu(self.fc3(x))
        x = self.drop(x)
        x = self.fc4(x)

        return x

class G_Model(nn.Module):
    def __init__(self, input_dim,first_division = 2):
        super(G_Model, self).__init__()
        # Encoder: affine function
        self.fc1 = nn.Linear(input_dim,input_dim//first_division)
        # self.bn1 = nn.BatchNorm1d(input_dim//first_division)
        self.fc2 = nn.Linear(input_dim//first_division, input_dim//(first_division*2))
        # self.bn2 = nn.BatchNorm1d(input_dim//(first_division*2))
        self.fc3 = nn.Linear(input_dim//(first_division*2), input_dim//(first_division*4))
        # Decoder: affine function
        self.fc4 = nn.Linear( input_dim//(first_division*4),input_dim//(first_division*2))
        # self.bn3 = nn.BatchNorm1d(input_dim//(first_division*2))
        self.fc5 = nn.Linear( input_dim//(first_division*2),input_dim//(first_division))
        # self.bn4 = nn.BatchNorm1d(input_dim//(first_division))
        self.fc6 = nn.Linear(input_dim//first_division, input_dim)
        self.sig = nn.Sigmoid()
        self.selu = nn.SELU()

    def forward(self, a):
        x = self.selu(self.fc1(a))
        # x = self.bn1(x)
        x = self.selu(self.fc2(x))
        # x = self.bn2(x)
        z = self.selu(self.fc3(x))
        x = self.selu(self.fc4(z))
        # x = self.bn3(x)
        x = self.selu(self.fc5(x))
        # x = self.bn4(x) 
        logits = self.fc6(x)
        mask = self.sig(logits)
        return mask


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x