import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report


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


input_size = 2  
hidden_size = 48
output_size = 1
num_layers = 2



data = pd.read_csv("cleaned_data5.csv")  
X_test = data[['accel_magnitude', 'gyro_magnitude']].values
y_test = data['hit'].astype(int).values


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)


def create_windows(X, window_size=50, step_size=5):
    X_windows = []
    for i in range(0, len(X) - window_size + 1, step_size):
        X_windows.append(X[i:i+window_size])
    return np.array(X_windows)

window_size = 150  
step_size = 5
X_test_windows = create_windows(X_test_scaled, window_size, step_size)


X_test_tensor = torch.tensor(X_test_windows, dtype=torch.float32)


model = LSTMModel(input_size, hidden_size, output_size, num_layers)


model.load_state_dict(torch.load("lstm_model4.pth"))
model.eval()  
print("Model loaded and ready for testing.")


with torch.no_grad():
    y_pred_prob = model(X_test_tensor).squeeze().cpu().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)  


y_test_windows = np.array([1 if np.sum(y_test[i:i+window_size]) > 0 else 0 
                           for i in range(0, len(y_test) - window_size + 1, step_size)])


accuracy = accuracy_score(y_test_windows, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test_windows, y_pred))

distinct_hits = 0
in_hit = False
predicted_hit_events = []
min_duration = 5  
min_separation = 5  

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


with open("test_predictions.txt", "w") as f:
    f.write("Predictions (1 = hit, 0 = no hit):\n")
    for i, pred in enumerate(y_pred):
        f.write(f"Window {i}: {pred}\n")

print("Predictions saved to test_predictions.txt")
