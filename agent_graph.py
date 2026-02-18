from langgraph.graph import StateGraph
from agent_state import SocraticState
from agents import SocraticAgents

def create_agent_graph():
    # Create the state for the graph
    state = StateGraph(SocraticState)

    # Create LLM instances and add to the graph
    agents = SocraticAgents(context_switch=False)

    state.add_node("arbiter", agents.arbiter_node())
    state.add_node("elenchus", agents.elenchus_node())
    state.add_node("aporia", agents.aporia_node())
    state.add_node("maieutics", agents.maieutics_node())
    state.add_node("dialectic", agents.dialectic_node())

    # Since arbiter is supposed to feed into which learning model, it is the starting point
    state.set_entry_point("arbiter")

    # From here, finish building out the rest of the agent node roadmap