import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.activations import sigmoid, relu, softmax
from utils.losses import cross_entropy_loss
from models.ann import ANN

# 1. Load the Iris dataset
df = pd.read_csv('data/iris.csv')
X = df.drop(columns=['species']).values
y = df['species'].values

# 2. Encode the labels (species names) to numeric values (0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert y_train and y_test to one-hot encoding
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_encoded_one_hot = one_hot_encode(y_encoded, 3)

# 3. Split data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_one_hot, test_size=0.33, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Define the neural network model
model = ANN(input_size=4, hidden_size1=16, hidden_size2=12, output_size=3)

# 5. Train the model
epochs = 1000
learning_rate = 0.001
batch_size = len(X_train)

# Store loss for plotting
losses = []

# Early stopping parameters
patience = 10  # Number of epochs with no improvement
best_loss = np.inf
patience_counter = 0

for epoch in range(epochs):
    # Forward pass
    output = model.forward(X_train)
    
    # Compute the loss (cross-entropy)
    loss = cross_entropy_loss(output, y_train)
    
    # Backward pass (gradient descent)
    model.backward(X_train, y_train, output, learning_rate)
    
    # Store the loss for plotting
    losses.append(loss)
    
    # Check if loss improved
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

# 6. Get predictions on the test set
test_output = model.forward(X_test)

# Convert the softmax output to predicted class (0, 1, 2)
preds = np.argmax(test_output, axis=1)

# 7. Calculate test accuracy
accuracy = np.mean(preds == np.argmax(y_test, axis=1))
print(f'Test Accuracy: {accuracy:.4f}')

# 8. Generate the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 9. Optionally, save the model weights for later use
model.save_weights('saved_weights.npz')

# 10. Plot the training loss curve
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
