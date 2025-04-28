import numpy as np

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    
    # If y_true is one-hot encoded, we use argmax to select the correct label
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Compute the cross-entropy loss
    log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-9)  # Added epsilon to avoid log(0)
    loss = np.sum(log_likelihood) / m
    return loss
