from .graph import app
from .models import UserMessage

if __name__ == "__main__":
    user = UserMessage(text="What is Azure OpenAI?")
    thread = {"configurable": {"thread_id": "demo"}}
    state = {"user": user}

    for event in app.stream(state, thread):
        print(event)

    final = app.get_state(thread).get("final")
    print("Final Answer:", final)