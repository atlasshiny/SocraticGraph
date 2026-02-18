from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class SocraticState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    mastery_score: float