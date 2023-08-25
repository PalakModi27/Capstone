import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulated data
destination_reviews = [
    "The hotel was amazing and the location was great.",
    "Terrible experience. The service was awful.",
    # ... more reviews
]

user_health_data = [
    "Allergic to nuts",
    "Vegetarian",
    # ... more health data
]

travel_scores = [5, 2]  # Simulated travel recommendation scores

# Split data into training and testing sets
reviews_train, reviews_test, health_train, health_test, scores_train, scores_test = train_test_split(
    destination_reviews, user_health_data, travel_scores, test_size=0.2, random_state=42
)
print(reviews_train)
print(reviews_test)
print(health_train)
print(health_test)
print(scores_train)
print(scores_test)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
#num labels is set to 1 indicating that the model is used for regression, that is detecting continuous value

# Preprocess data
def preprocess_data(reviews, health_data, tokenizer):
    inputs = []
    for review, health in zip(reviews, health_data):
        text = f"{review} User's health: {health}"
        print(text)
        inputs.append(tokenizer.encode(text, add_special_tokens=True))
    return torch.tensor(inputs)
# The review and health data are combined into a single input string and encoded using the tokenizer. 

train_inputs = preprocess_data(reviews_train, health_train, tokenizer)
print("Train Input:", train_inputs)
train_labels = torch.tensor(scores_train, dtype=torch.float32)
print("Train Label:", train_labels)
test_inputs = preprocess_data(reviews_test, health_test, tokenizer)
print("Test Input:",test_inputs)
test_labels = torch.tensor(scores_test, dtype=torch.float32)
print("Test Label:", train_inputs)
#The corresponding travel recommendation scores are also converted to PyTorch tensors.
#pytorch tensors are like numpy arrays

# Fine-tuning
# AdamW is a stochastic optimization method
# Setting up the MSE loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#print("Optimizer:",optimizer)
#optimizing the model parameters
loss_fn = torch.nn.MSELoss()

batch_size = 8
num_epochs = 3

train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
#allows you to preload your dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
# The loss is calculated using MSE and backpropagation is performed to update the model's parameters. The average loss for the epoch is printed.
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_labels in train_dataloader:
        optimizer.zero_grad()   #optimizer set to zero
        outputs = model(batch_inputs).logits.squeeze()
        loss = loss_fn(outputs, batch_labels)         #loss function is used to determine how good the model is performing
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}: Average Loss {average_loss:.4f}")

# ... (previous code)

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(test_inputs).logits.squeeze()
    print("Predictions:",predictions)

    # Calculate Mean Squared Error using PyTorch functions
    mse = torch.mean((predictions - test_labels) ** 2)
    rmse = torch.sqrt(mse)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")
    print(f"Root Mean Squared Error on Test Set: {rmse:.4f}")

    # Additional stats
    avg_actual_score = torch.mean(test_labels).item()
    avg_predicted_score = torch.mean(predictions).item()

# ... (previous code)

# Calculate the correlation between predictions and test labels
# The code calculates the Pearson correlation coefficient between the predicted scores and the test labels.
predictions_mean = torch.mean(predictions)
test_labels_mean = torch.mean(test_labels)

numerator = torch.sum((predictions - predictions_mean) * (test_labels - test_labels_mean))
denominator_predictions = torch.sqrt(torch.sum((predictions - predictions_mean) ** 2))
denominator_test_labels = torch.sqrt(torch.sum((test_labels - test_labels_mean) ** 2))

correlation = numerator / (denominator_predictions * denominator_test_labels)

# Convert the correlation to a Python scalar value
correlation_value = correlation.item()

print(f"Avg Actual Score: {avg_actual_score:.2f}")
print(f"Avg Predicted Score: {avg_predicted_score:.2f}")
print(f"Correlation: {correlation_value:.2f}")
