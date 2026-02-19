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

    # Arbiter decides next agent
    state.add_edge("arbiter", "elenchus", lambda state: state["next_agent"] == "elenchus")
    state.add_edge("arbiter", "maieutics", lambda state: state["next_agent"] == "maieutics")

    # Elenchus can lead to aporia, maieutics, or dialectic
    state.add_edge("elenchus", "aporia", lambda state: state.get("route") == "aporia")
    state.add_edge("elenchus", "maieutics", lambda state: state.get("route") == "maieutics")
    state.add_edge("elenchus", "dialectic", lambda state: state.get("route") == "dialectic")

    # Aporia can lead to maieutics or dialectic
    state.add_edge("aporia", "maieutics", lambda state: state.get("route") == "maieutics")
    state.add_edge("aporia", "dialectic", lambda state: state.get("route") == "dialectic")

    # Maieutics can lead to dialectic or arbiter
    state.add_edge("maieutics", "dialectic", lambda state: state.get("route") == "dialectic")
    state.add_edge("maieutics", "arbiter", lambda state: state.get("route") == "arbiter")

    # Dialectic can end or loop back
    state.add_edge("dialectic", "arbiter", lambda state: state.get("mastery_score", 0.0) < 0.9)
    # Terminal node: end if mastery_score >= 0.9
    state.add_edge("dialectic", END, lambda state: state.get("mastery_score", 0.0) >= 0.9)

    return state.compile()