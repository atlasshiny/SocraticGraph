from langgraph.graph import StateGraph, END
from agent_state import SocraticState
from agents import SocraticAgents

def create_agent_graph(agents: SocraticAgents):
    """
    Construct the agent graph, adding nodes and conditional edges for Socratic dialogue.
    Args:
        agents (SocraticAgents): The collection of agent node callables.
    Returns:
        Compiled graph ready for execution.
    """
    # Create the state for the graph
    state = StateGraph(SocraticState)

    state.add_node("arbiter", agents.arbiter_node)
    state.add_node("elenchus", agents.elenchus_node)
    state.add_node("aporia", agents.aporia_node)
    state.add_node("maieutics", agents.maieutics_node)
    state.add_node("dialectic", agents.dialectic_node)

    # Since arbiter is supposed to feed into which learning model is going to be used, it is the starting point
    state.set_entry_point("arbiter")

    # Arbiter conditional edge: next_agent determines next node
    state.add_conditional_edges("arbiter", lambda state: state["next_agent"])

    # After the chosen agent (elenchus/aporia/maieutics) runs, always go to dialectic for evaluation
    state.add_edge("elenchus", "dialectic")
    state.add_edge("aporia", "dialectic")
    state.add_edge("maieutics", "dialectic")

    # Dialectic always ends the current graph execution.
    # The outer while-loop in main.py re-prompts the user and re-invokes the graph with updated history, so the user can respond each cycle.
    state.add_edge("dialectic", END)

    return state.compile()