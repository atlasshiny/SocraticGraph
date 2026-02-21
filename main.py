from pathlib import Path

from agents import SocraticAgents
from agent_graph import create_agent_graph
from langchain_core.messages import AIMessage, HumanMessage
from history import (
    HISTORY_ENABLED_DEFAULT,
    HISTORY_FILE_NAME,
    CONTEXT_TOKEN_BUDGET,
    cap_messages,
    load_history,
    save_history,
    reset_history,
)

def main():
    """
    Main entry point for the Socratic agent loop. Handles user input, runs the agent graph, and displays output.
    """
    agents = SocraticAgents(context_switch=True)

    loop = create_agent_graph(agents=agents)
    history_path = Path(__file__).with_name(HISTORY_FILE_NAME)
    history_enabled = HISTORY_ENABLED_DEFAULT
    history = load_history(history_path) if history_enabled else []

    print("Toggle message history with 'history on/off' or reset with 'reset'")
    while True:
        user_input = input("User: ")

        # Add exit to loop
        if user_input.lower() in ["quit", "exit"]:
            break

        # runtime commands to control history behaviour
        cmd = user_input.strip().lower()
        if cmd in ("history off",):
            history_enabled = False
            print("Persistent history disabled for this session.")
            continue
        if cmd in ("history on",):
            history_enabled = True
            history = load_history(history_path)
            print(f"Persistent history enabled. Loaded {len(history)} messages.")
            continue
        if cmd in ("reset", "reset history", "history reset"):
            history = []
            try:
                reset_history(history_path)
            except Exception:
                pass
            print("History reset; conversation memory cleared.")
            continue

        user_message = HumanMessage(content=user_input)
        # Trim the combined history + current user message to the token budget
        turn_messages = cap_messages(history + [user_message], CONTEXT_TOKEN_BUDGET)
        agent_messages = []

        # Stream the graph execution
        finished = False
        for event in loop.stream({"messages": turn_messages}):
            for node_name, output in event.items():
                # Print messages from the agents so the user can see the communication
                if "messages" in output:
                    node_messages = output["messages"]
                    if node_messages:
                        for node_message in node_messages:
                            if isinstance(node_message, AIMessage):
                                agent_messages.append(node_message)
                        print(f"\n[{node_name.upper()}]: {node_messages[-1].content}")
                # Print raw arbiter output for debugging when available
                if "arbiter_raw" in output:
                    print(f"[ARBITER_RAW]: {output['arbiter_raw']}")
                if "mastery_score" in output:
                    score = output['mastery_score']
                    print(f"--- Current Mastery Score: {score} ---")
                    if score >= 0.9:
                        print("*** Mastery threshold reached (>= 0.9). Interaction finished. You may start a new topic. ***")
                        finished = True
                        break
            if finished:
                break

        # Persist only when history is enabled
        if history_enabled:
            history = history + [user_message] + agent_messages
            save_history(history_path, history)

if __name__ == "__main__":
    main()