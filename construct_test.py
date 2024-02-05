import json

prefix = (
    "Task: Emotion-Flip Recognition (EFR). Objective: Identify the trigger utterance(s) in a multi-party "
    "conversation that causes a change in emotions. Note: Each utterance is labeled with a speaker and their emotion. "
    "Demonstration: In a conversation - "
    "1.Chandler: 'I was the point person on the KL-5 to GR-6 system transition.' (neutral) "
    "2.The Interviewer: 'You must've had your hands full.' (neutral) "
    "3.Chandler: 'That I did.' (neutral) "
    "4.The Interviewer: 'Let's talk about your duties.' (neutral) "
    "5.Chandler: 'My duties? All right.' (surprise) "
    "The trigger for the emotion flip is utterance 4 by The Interviewer. "
    "Now, analyze the following dialogue and identify the trigger utterance(s): "
)

test_data_path = "MELD_test_efr.json"
test_save_path = "./test_meld_efr.json"

with open(test_data_path, "r") as f:
    test_data = json.load(f)

test_conversations = []

for dialogue in test_data:
    utterances = ""  # Reset for each new dialogue
    for i, (speaker, emotion, utterance) in enumerate(
        zip(dialogue["speakers"], dialogue["emotions"], dialogue["utterances"])
    ):
        formatted_utterance = f"{i + 1}.{speaker}: '{utterance}' ({emotion}) "
        utterances += formatted_utterance

    context = f"{prefix}Dialogue: {utterances}"
    test_conversation = {
        "context": context,
        "target": "",
    }  # Include an empty 'target' field
    test_conversations.append(test_conversation)

print("Test dataset creation success")
json_test_data = json.dumps(test_conversations, indent=4)
with open(test_save_path, "w") as f:
    f.write(json_test_data)
