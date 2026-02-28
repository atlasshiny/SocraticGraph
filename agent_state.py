from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class SocraticState(TypedDict):
    """
    State dictionary for tracking the Socratic agent loop, including messages, routing, and mastery score.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    mastery_score: float
    arbiter_raw: str
    dialectic_raw: str