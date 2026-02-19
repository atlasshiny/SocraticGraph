from agents import SocraticAgents
from agent_graph import create_agent_graph
from langchain_core.messages import HumanMessage

def main():
    agents = SocraticAgents(context_switch=False)

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
                if "mastery_score" in output:
                    print(f"--- Current Mastery Score: {output['mastery_score']} ---")

if __name__ == "__main__":
    main()