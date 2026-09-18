import boto3
import json
import os

from dotenv import load_dotenv

load_dotenv()

# Connect to Bedrock and choose the model
client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
model_id = os.getenv("BEDROCK_MODEL_ID")

# Give the chatbot a personality
system_prompt = [{"text": "You are a customer support assistant."
" Answer questions about Account, Orders and Billing"}]

HISTORY_FILE = "conversation_history.json"

try:
    with open(HISTORY_FILE, "r") as f:
        messages = json.load(f)
    # print("Loaded previous conversation.")
except (FileNotFoundError, json.JSONDecodeError):
    messages = []

print("Chatbot ready! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    # Add the user's message to conversation history
    messages.append({"role": "user", "content": [{"text": user_input}]})

    # Send the full conversation to Bedrock with guardrail protection
    response = client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompt,
        inferenceConfig={
            "temperature": 0.7,
            "topP": 0.9,
            "maxTokens": 512
        },
        guardrailConfig={
        "guardrailIdentifier": os.getenv("GUARDRAIL_ID"),
        "guardrailVersion": os.getenv("GUARDRAIL_VERSION"),
        "trace": "enabled"
        }
    )

      # Save the response and print it
    assistant_message = response["output"]["message"]
    messages.append(assistant_message)

    print(f"Bot: {assistant_message['content'][0]['text']}")
    # print(response.get("trace"))

    # --- NEW: write the updated list to disk after every turn ---
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f, indent=2) 
         # this line has nothing to do with Bedrock — it's 
         # just local file saving
