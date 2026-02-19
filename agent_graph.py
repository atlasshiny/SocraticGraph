from langgraph.graph import StateGraph, END
from agent_state import SocraticState
from agents import SocraticAgents

def create_agent_graph(agents: SocraticAgents):
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

    # Elenchus conditional edge: route determines next node
    state.add_conditional_edges("elenchus", lambda state: state.get("route"))

    # Aporia conditional edge: route determines next node
    state.add_conditional_edges("aporia", lambda state: state.get("route"))

    # Maieutics conditional edge: route determines next node
    state.add_conditional_edges("maieutics", lambda state: state.get("route"))

    # Dialectic conditional edge: mastery_score determines next node or END
    state.add_conditional_edges(
        "dialectic",
        lambda state: END if state.get("mastery_score", 0.0) >= 0.9 else "arbiter"
    )

    return state.compile()