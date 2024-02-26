from langchain.memory import ChatMessageHistory


def create_history(messages,query):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])

        else:
            history.add_ai_message(message["content"])
    history.add_user_message(query["content"])
    return history