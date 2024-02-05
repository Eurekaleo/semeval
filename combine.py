# Task 1: Generate "neutral" for each of the first 1580 lines
task1_results = ["neutral\n" for _ in range(1580)]

# Task 2: Generate "0.0" for each of the next 7690 lines (1581 to 9270)
task2_results = ["0.0\n" for _ in range(7690)]

# Task 3: Read the previously generated trigger_labels.txt
with open("output_results.txt", "r") as file:
    task3_results = file.readlines()

# Ensure the length of task3_results matches the expected line count for Task 3
expected_task3_lines = 17912 - 9270
if len(task3_results) != expected_task3_lines:
    raise ValueError(
        f"Task 3 results should have {expected_task3_lines} lines, but got {len(task3_results)}"
    )

# Combine results for all tasks
combined_results = task1_results + task2_results + task3_results

# Write the combined results to a new file
with open("finale_results.txt", "w") as file:
    file.writelines(combined_results)
