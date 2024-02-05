import json

# Load the test dataset
with open("MELD_test_efr.json", "r") as file:
    test_set = json.load(file)

# Load the model predictions
with open("newtrain.json", "r") as file:
    predictions = json.load(file)


# Function to extract trigger indices from prediction string
def extract_indices(prediction):
    # Extract the numeric part and split by commas
    indices_str = prediction.split(":")[1].strip()
    # If indices string is empty, return an empty list
    if not indices_str:
        return []
    # Otherwise, convert each index to zero-based index
    return [int(index) - 1 for index in indices_str.split(",")]


# Process predictions to match the test set format
results = []
for i, episode in enumerate(test_set):
    # Initialize all utterances as non-trigger (0.0)
    episode_results = ["0.0"] * len(episode["utterances"])
    # Get the predicted trigger indices
    trigger_indices = extract_indices(predictions[i])
    # Mark the trigger utterances
    for index in trigger_indices:
        if index >= 0 and index < len(episode_results):  # Check index bounds
            episode_results[index] = "1.0"
    # Add to overall results
    results.extend(episode_results)

# Rest of the code remains the same


# Join all results into a single string for file output
output_content = "\n".join(results)

# Save to a file or use as needed
with open("output_results.txt", "w") as output_file:
    output_file.write(output_content)
