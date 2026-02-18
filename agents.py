from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

class SocraticAgents():
    def __init__(self, context_switch: bool = True):
        if context_switch:
            arbiter_model = "phi4-mini:3.8b-q4_K_M"
            elenchus_model = "mistral:7b-instruct-v0.3-q3_K_S"
            aporia_model = "llama3.1:8b-instruct-q2_K"
            maieutics_model = "llama3.1:8b-instruct-q2_K"
            dialectic_model = "phi4-mini:3.8b-q4_K_M"
        else:
            arbiter_model = elenchus_model = aporia_model = maieutics_model = dialectic_model = "llama3.1:8b-instruct-q2_K"

        # The Orchestrator: High precision for logical routing and intent classification. t=0 for deterministic pathing.
        self.arbiter_llm = ChatOllama(model=arbiter_model, temperature=0.0)

        # The Adversary: Tuned for logical rigor and cross-examination. Low temperature for solid, factual responses
        self.elenchus_llm = ChatOllama(model=elenchus_model, temperature=0.1)

        # The Puzzler: Slightly higher temperature (0.7) to allow for creative analogies and the generation of 'productive doubt' paradoxes.
        self.aporia_llm = ChatOllama(model=aporia_model, temperature=0.7)

        #The Midwife: Balanced temperature (0.5) for clear, constructive scaffolding and analogy-based knowledge synthesis.
        self.maieutics_llm = ChatOllama(model=maieutics_model, temperature=0.5)

        #The Auditor: Evaluates state transitions and verifies concept mastery. t=0 for objectiveness and determinism.
        self.dialectic_llm = ChatOllama(model=dialectic_model, temperature=0.0)

        # The system "objective" prompts
        self.prompts = {
            "arbiter": "Analyze the conversation. Route to 'elenchus' if the user is overconfident, or 'maieutics' if they need help.",
            "elenchus": "Find logical contradictions in the user's statement. Be sharp and persistent.",
            "aporia": "Create a sense of wonder and doubt. Use paradoxes to show why this topic is difficult.",
            "maieutics": "Use analogies to help the user discover the truth themselves. Do not give the answer.",
            "dialectic": "Evaluate the user's progress. Assign a mastery score from 0.0 to 1.0."
        }