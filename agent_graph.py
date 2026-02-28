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

    # Dialectic evaluates mastery: loop back to arbiter if below threshold, else finish.
    def dialectic_router(state):
        if state.get("mastery_score", 0) >= 0.9:
            return END
        return "arbiter"

    state.add_conditional_edges("dialectic", dialectic_router)

    return state.compile()