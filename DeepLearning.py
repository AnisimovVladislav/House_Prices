import torch.nn as nn

def nn_model(n_features):
    return nn.Sequential(
        nn.Linear(n_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        
        nn.Linear(128, 1),
    )
    
    

    
