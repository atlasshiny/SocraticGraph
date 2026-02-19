from agents import SocraticAgents
from agent_graph import create_agent_graph
from langchain_core.messages import HumanMessage

def main():
    """
    Main entry point for the Socratic agent loop. Handles user input, runs the agent graph, and displays output.
    """
    agents = SocraticAgents(context_switch=True)

    loop = create_agent_graph(agents=agents)

    while True:
        user_input = input("User: ")

        # Add exit to loop
        if user_input.lower() in ["quit", "exit"]:
            break

        # Stream the graph execution
        for event in loop.stream({"messages": [HumanMessage(content=user_input)]}):
            for node_name, output in event.items():
                # Print messages from the agents so the user can see the communication
                if "messages" in output:
                    print(f"\n[{node_name.upper()}]: {output['messages'][-1].content}")
                # Print raw arbiter output for debugging when available
                if "arbiter_raw" in output:
                    print(f"[ARBITER_RAW]: {output['arbiter_raw']}")
                if "mastery_score" in output:
                    print(f"--- Current Mastery Score: {output['mastery_score']} ---")

if __name__ == "__main__":
    main()