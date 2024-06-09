import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from flask import Flask, request, jsonify, render_template
import random

class Blockchain:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

def generate_transaction(index, malicious=False):
    amount = np.random.rand() * 100
    fee = np.random.rand()
    size = np.random.randint(250, 1000)

    if malicious:
        amount *= 10

    is_malicious = 0
    if amount > 50 and fee < 0.01:
        is_malicious = 1
    elif size > 750 and fee < 0.01:
        is_malicious = 1
    elif amount < 10 and fee > 0.5:
        is_malicious = 0
    elif amount > 50 and fee > 0.1:
        is_malicious = 0
    elif size < 500 and fee > 0.1:
        is_malicious = 0

    if malicious:
        is_malicious = 1

    return {
        'index': index,
        'amount': amount,
        'sender': f"user{np.random.randint(1, 100)}",
        'receiver': f"user{np.random.randint(1, 100)}",
        'fee': fee,
        'size': size,
        'is_malicious': is_malicious
    }

def extract_features(transaction):
    return [
        transaction['amount'],
        len(transaction['sender']),
        len(transaction['receiver']),
        transaction['fee'],
        transaction['size']
    ]

# Initialize blockchain
blockchain = Blockchain()

# Generate and label transactions
transactions = [generate_transaction(i) for i in range(90)]
malicious_transactions = [generate_transaction(i, malicious=True) for i in range(90, 100)]
transactions.extend(malicious_transactions)

# Feature extraction
transaction_features = np.array([extract_features(tx) for tx in transactions])
labels = np.array([tx['is_malicious'] for tx in transactions]).reshape(-1, 1)

# Combine features and labels for resampling
data_with_labels = np.hstack((transaction_features, labels))

# Balance the dataset by oversampling the minority class
minority_class = data_with_labels[data_with_labels[:, -1] == 1]
majority_class = data_with_labels[data_with_labels[:, -1] == 0]

if len(minority_class) > 0:
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    balanced_data = np.vstack((majority_class, minority_upsampled))
else:
    balanced_data = data_with_labels

# Shuffle the balanced data
np.random.shuffle(balanced_data)

# Separate features and labels
transaction_features_balanced = balanced_data[:, :-1]
labels_balanced = balanced_data[:, -1].reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
transaction_features_scaled = scaler.fit_transform(transaction_features_balanced)
data_with_labels = np.hstack((transaction_features_scaled, labels_balanced))

# LSTM model setup
seq_length = 10
X = []
y = []

for i in range(len(data_with_labels) - seq_length):
    X.append(data_with_labels[i:i+seq_length, :-1])
    y.append(data_with_labels[i+seq_length, -1])

X = np.array(X)
y = np.array(y)

# Train LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=20, batch_size=10, validation_split=0.2)

# Evaluate LSTM model
y_pred = (model.predict(X) > 0.5).astype("int32")
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Save metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

# Train SOM model
som_grid_size = int(np.sqrt(5 * np.sqrt(X.shape[0])))  # Dynamically choose grid size
som = MiniSom(x=som_grid_size, y=som_grid_size, input_len=X.shape[2] * seq_length, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X.reshape(-1, X.shape[2] * seq_length))
som.train_random(X.reshape(-1, X.shape[2] * seq_length), 100)

# Save models and scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
model.save('lstm_model.h5')

# Plot LSTM training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('static/loss_plot.png')
plt.close()

# Plot SOM result
plt.figure(figsize=(7, 7))
for i, x in enumerate(X.reshape(-1, X.shape[2] * seq_length)):
    w = som.winner(x)
    plt.text(w[0] + 0.5, w[1] + 0.5, str(int(y[i])), color=plt.cm.rainbow(y[i] / 2), fontdict={'weight': 'bold', 'size': 11})
plt.xlim([0, som.get_weights().shape[0]])
plt.ylim([0, som.get_weights().shape[1]])
plt.title('SOM Clusters')
plt.grid()
plt.savefig('static/som_plot.png')
plt.close()

# Correlation heatmap for transaction features
feature_labels = ['Amount', 'Sender Length', 'Receiver Length', 'Fee', 'Size', 'Target']
data_and_target = pd.concat([pd.DataFrame(transaction_features_balanced, columns=feature_labels[:-1]), 
                             pd.DataFrame(labels_balanced, columns=[feature_labels[-1]])], axis=1)
var_corr = data_and_target.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap of Transaction Features')
plt.savefig('static/heatmap.png')
plt.close()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    transaction = {
        'amount': float(data['amount']),
        'sender': data['sender'],
        'receiver': data['receiver'],
        'fee': float(data['fee']),
        'size': int(data['size'])
    }
    
    features = np.array([[
        transaction['amount'],
        len(transaction['sender']),
        len(transaction['receiver']),
        transaction['fee'],
        transaction['size']
    ]])

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    features_scaled = scaler.transform(features)
    
    features_scaled = np.expand_dims(np.tile(features_scaled, (seq_length, 1)), axis=0)
    
    prediction = model.predict(features_scaled)
    
    result = 'Malicious' if prediction > 0.5 else 'Honest'
    return jsonify(prediction=result)

@app.route('/metrics')
def show_metrics():
    formatted_metrics = {k: f"{v * 100:.2f}%" for k, v in metrics.items()}
    return render_template('metrics.html', metrics=formatted_metrics)

@app.route('/visualisation')
def visualisation():
    return render_template('visualisation.html')

@app.route('/transactions')
def show_transactions():
    shuffled_transactions = random.sample(transactions, len(transactions))
    return render_template('transactions.html', transactions=shuffled_transactions)

@app.route('/transaction/<int:transaction_id>')
def transaction_detail(transaction_id):
    transaction = next(tx for tx in transactions if tx['index'] == transaction_id)
    return render_template('transaction_detail.html', transaction=transaction)

if __name__ == '__main__':
    app.run(debug=True)
