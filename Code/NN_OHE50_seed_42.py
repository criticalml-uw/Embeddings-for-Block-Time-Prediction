# -*- coding: utf-8 -*-
"""Pytorch_Embeddings_all_data_train_val_test.ipynb

"MAIN CHANGE IS THIS DOES NOT HAVE EMBEDDING LAYER ANYMORE"

Changes from the original - 
1. We will remove the dropout (self.emb_drop and self.drops) and batch normalization layers (self.bn1, self.bn2, and self.bn3). 
2. We will modify the forward function to have just the linear layers followed by a ReLU activation
3. Instead of the embedding layer we ohe each categrical variable and then concatenate them , then apply a linear + relu on it ( reducing the output to 1461(total unique categorical states )//(1)

Main changes from arch 4 --> output features instead of 0.75 is now 1 i.e no compression


(This takes more than 25 hrs to run )

"""

# ------------ IMPORT LIBRARIES --------------
import os
import math
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


# -- importing other modules 
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


#---- Setting the random seed for reproducibility---------
SEED = 42

# For NumPy
np.random.seed(SEED)

# For PyTorch
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
# ---- Load data for a seed -----------------
seed=42  # For data 
data_dir = "/home/aniketb/scratch/data/BTS_processed_2018_by_seed"

# Load train data
with open(f"{data_dir}/{seed}/train_data.pkl", "rb") as f:
    X_train, y_train = pickle.load(f)

# Load validation data
with open(f"{data_dir}/{seed}/val_data.pkl", "rb") as f:
    X_val, y_val = pickle.load(f)

# Load test data
with open(f"{data_dir}/{seed}/test_data.pkl", "rb") as f:
    X_test, y_test = pickle.load(f)
        
#------Result File ----------
results_file = f"BTS/Pytorch_embeddings_jobs/output_paper1_results/emb_0_arch_5/results_seed_{seed}.txt"  

#---- Summary Writer for logs 
# The base directory for tensorboard logs
base_dir = f"/home/aniketb/scratch/saved/logs_output_paper1/emb_0_arch_5/{seed}"
log_dir = f"{base_dir}/"
writer = SummaryWriter(log_dir=log_dir)
    
    
#--- Features --    
categorical_features= ['Quarter',
'Month' ,
'DayofMonth' ,
'DayOfWeek',
'Reporting_Airline',
#'Tail_Number',
#'Flight_Number_Reporting_Airline',
'OriginAirportSeqID',
'Origin',
'DestAirportSeqID',
'Dest',
'Weekend',
'IATASeason'
]
numerical_features = [
 'FlightSequenceTails',
 'sin_CRSDepTime',
 'OriginLongitude',
 'cos_CRSDepTime',
 'sin_CRSArrTime',
 'OriginLatitude',
 'cos_CRSArrTime',
 'DestLongitude',
 'DestLatitude',
 'Distance',
 'CRSElapsedTime']
target='BlockTime'


# Convert data into PyTorch tensors
X_cat_train = torch.tensor(X_train[categorical_features].values, dtype=torch.int64)
X_cat_val = torch.tensor(X_val[categorical_features].values, dtype=torch.int64)
X_cat_test = torch.tensor(X_test[categorical_features].values, dtype=torch.int64)
X_cont_train = torch.tensor(X_train[numerical_features].values, dtype=torch.float32)
X_cont_val = torch.tensor(X_val[numerical_features].values, dtype=torch.float32)
X_cont_test = torch.tensor(X_test[numerical_features].values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

"""# Define the Network Architecture"""

# --- 1. Define Network Architecture ---
# After the ohe encoding the linear layer reduces to half the dimensions 
class TabularModel(nn.Module):
    def __init__(self, cat_dims, n_cont):  # cat_dims: List of unique values per categorical column
        super(TabularModel, self).__init__()
        
        self.cat_dims = cat_dims
        self.n_cat = sum(cat_dims)  # Total number of unique values across all categorical columns
        self.n_cont = n_cont
        
        # Linear + ReLU for categorical features
        self.lin_cat = nn.Linear(self.n_cat, self.n_cat * 1)  # Output size is 3/4 th of the input size
        
        # Fully connected layers
        self.lin1 = nn.Linear(self.n_cat * 1 + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 1)

    def one_hot_encode(self, x_cat):
        # One-hot encode each categorical feature and then concatenate them
        return torch.cat([F.one_hot(x_cat[:, i], self.cat_dims[i]) for i in range(x_cat.size(1))], dim=1)

    def forward(self, x_cat, x_cont):
        x_cat = self.one_hot_encode(x_cat).float()  # One-hot encoding the categorical features
        x_cat = F.relu(self.lin_cat(x_cat))  # Linear + ReLU for categorical features
        x_cont = x_cont.float()  # Ensure x_cont is also a float
        x = torch.cat([x_cat, x_cont], 1)  # Concatenation of one-hot encoded categorical data and continuous data
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


cat_dims = list(X_train[col].nunique() for col in categorical_features)
n_cont = len(numerical_features)

model = TabularModel(cat_dims, n_cont)
print("Model architecture:\n", model)

#--------- TENSORBOARD GRAPH 

dummy_input_cat = torch.zeros(1, len(categorical_features), dtype=torch.int64)
dummy_input_cont = torch.zeros(1, len(numerical_features), dtype=torch.float32)
writer.add_graph(model, (dummy_input_cat, dummy_input_cont))

#--------------

"""# Prepare Data for training"""

# --- 2. Prepare Data Loaders ---
train_ds = TensorDataset(X_cat_train, X_cont_train, y_train_tensor)
val_ds = TensorDataset(X_cat_val, X_cont_val, y_val_tensor)
test_ds = TensorDataset(X_cat_test, X_cont_test, y_test_tensor)

#Define a empty list to store results 
results = []

# Define different batch sizes and learning rates you want to experiment with
batch_sizes = [32, 64, 128] 
learning_rates = [0.001, 0.005, 0.0001]

for batch in batch_sizes:
    for lr in learning_rates:

        # Create a unique log directory for this combination of hyperparameters
        log_dir = f"{base_dir}/lr_{lr}_batch_{batch}"
        writer = SummaryWriter(log_dir=log_dir)

        # DataLoader with the current batch size
        train_loader = DataLoader(dataset=train_ds, batch_size=batch, shuffle=True,
                                  worker_init_fn=lambda _: np.random.seed(SEED))
        val_loader = DataLoader(dataset=val_ds, batch_size=batch, shuffle=False)
        test_loader = DataLoader(dataset=test_ds, batch_size=batch, shuffle=False)
        
        # Create instance of the neural network
        model = TabularModel(cat_dims, n_cont)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define loss function and optimizer with the current learning rate
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the model with early stopping using the validation set
        num_epochs = 250
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')  # Track the best validation loss
        best_model_state = None  # Track the state of the best model

        patience = 10  # Define the patience value for early stopping
        count = 0  # Counter for tracking patience
        early_stopping = False  # Flag for early stopping

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for x1, x2, y in train_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                optimizer.zero_grad()

                outputs = model(x1, x2)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)

            train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for x1_val, x2_val, y_val in val_loader:
                    x1_val, x2_val, y_val = x1_val.to(device), x2_val.to(device), y_val.to(device)

                    val_outputs = model(x1_val, x2_val)
                    val_batch_loss = criterion(val_outputs, y_val).item()
                    val_loss += val_batch_loss * len(y_val)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Log losses to TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)

            # Check if the current validation loss is better than the best validation loss so far
            if val_loss < best_val_loss:
                count = 0  # Reset the counter
                best_val_loss = val_loss
                best_model_state = model.state_dict()
            else:
                count += 1  # Increment the counter

                # If the counter exceeds the patience value, perform early stopping
                if count >= patience:
                    early_stopping = True
                    print("Early stopping triggered!")
                    break

        # Test the model
        model.load_state_dict(best_model_state)
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for x1_test, x2_test, y_test in test_loader:
                x1_test, x2_test, y_test = x1_test.to(device), x2_test.to(device), y_test.to(device)

                test_outputs = model(x1_test, x2_test)
                test_batch_loss = criterion(test_outputs, y_test).item()
                test_loss += test_batch_loss * len(y_test)

        test_loss /= len(test_loader.dataset)
        print(f"Test Loss for LR = {lr} and Batch Size = {batch}: {test_loss:.4f}")

        # Log the hyperparameters and results
        writer.add_hparams({
            'lr': lr,
            'batch_size': batch,
        }, {
            'train_loss': train_loss,  # Final training loss for this run
            'val_loss': val_loss,      # Final validation loss for this run
            'test_loss': test_loss,    # Final test loss for this run
        })

        # Store the results for this combination in a dictionary
        results_dict = {
            'Learning Rate': lr,
            'Batch Size': batch,
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Test Loss': test_loss
        }
            
        # Append this dictionary to the results list
        results.append(results_dict)

# Close the SummaryWriter
writer.close()

# Print results 
for res in results:
    lr = res['Learning Rate']
    batch = res['Batch Size']
    train_loss = res['Train Loss']
    val_loss = res['Validation Loss']
    test_loss = res['Test Loss']
    
    print(f"{lr}\t{batch}\t{train_loss:.4f}\t{val_loss:.4f}\t{test_loss:.4f}")

# Write the results in a text file 
# Now, write the collected results to a text file
with open(results_file, 'w') as file:
    # Save the seed of the data
    file.write("Data Seed :\n")
    file.write(str(seed))
    file.write("\n\n")
    
    # Save the model architecture
    file.write("Model Architecture:\n")
    file.write(str(model))  # Convert the model architecture to string and write to file
    file.write("\n\n")
    
       
    # Results
    file.write('Results:\n')
    file.write('Learning Rate\tBatch Size\tTrain Loss\tValidation Loss\tTest Loss\n')
    file.write('-------------------------------------------------------------------------------------\n')
    
    # Loop through the results and write each one to the file
    for res in results:
        file.write(f"{res['Learning Rate']}\t{res['Batch Size']}\t{res['Train Loss']:.4f}\t{res['Validation Loss']:.4f}\t{res['Test Loss']:.4f}\n")












