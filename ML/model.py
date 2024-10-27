import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import glob


data = pd.read_csv('ML/alldata_filtered.csv')  

data = data.dropna()






X = data[['x', 'y', 'z', 'accel_magnitude', 'gyro_magnitude']]
y = data['hit'].astype(int)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_windows(X, y, window_size=250, step_size=50):
    X_windows, y_windows = [], []
    for i in range(0, len(X) - window_size + 1, step_size):
        X_window = X[i:i+window_size]
        y_window = y[i:i+window_size]
        X_windows.append(X_window)
        
        
        y_windows.append(1 if y_window.sum() > 0 else 0)
    return np.array(X_windows), np.array(y_windows)

window_size = 150  
step_size = 5     
X_windows, y_windows = create_windows(X_scaled, y, window_size, step_size)


X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)























batch_size = 32
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  
        out = self.fc(out)
        return self.sigmoid(out)  



input_size = X_train.shape[2]
hidden_size = 48
output_size = 1
num_layers = 2
learning_rate = 0.0005
num_epochs = 11


model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.BCELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        
        y_batch = y_batch.float().view(-1, 1)

        
        outputs = model(X_batch)

        
        if torch.isnan(outputs).any():
            print("NaN detected in outputs. Stopping training.")
            break

        
        loss = criterion(outputs, y_batch)
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    
    if torch.isnan(outputs).any():
        print("Stopping training due to NaNs in outputs.")
        break


model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).squeeze().cpu().numpy()
    y_pred = (y_pred_prob > 0.7).astype(int)  


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


distinct_hits = 0
in_hit = False
predicted_hit_events = []
min_duration = 3  
min_separation = 3  

count = 0  
last_hit_end = -min_separation  

for i in range(len(y_pred)):
    if y_pred[i] == 1:
        if count == 0:  
            hit_start = i
        count += 1
    else:
        
        if count >= min_duration and (i - last_hit_end) >= min_separation:
            distinct_hits += 1
            predicted_hit_events.append(hit_start)  
            last_hit_end = i  
        
        count = 0


if count >= min_duration and (len(y_pred) - last_hit_end) >= min_separation:
    distinct_hits += 1
    predicted_hit_events.append(hit_start)


print("Detected hit start times (indices):", predicted_hit_events)
print(f"Total distinct hit events predicted: {distinct_hits}")


output_filename = "y_pred_output.txt"


with open(output_filename, "w") as f:
    f.write("y_pred output:\n")
    for i, pred in enumerate(y_pred):
        f.write(f"Index {i}: {pred}\n")

print(f"y_pred sequence has been written to {output_filename}")

torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved after training.")


