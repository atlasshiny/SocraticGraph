from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from agent_state import SocraticState 

class SocraticAgents():
    """
    A collection of agent nodes for Socratic dialogue, each representing a different role in the learning process.
    """
    def __init__(self, context_switch: bool = True):
        """
        Initialize all agent LLMs and their prompts. Optionally switch context for model selection.
        """
        ollama_backend = "cuda"

        if context_switch:
            arbiter_model = "phi4-mini:3.8b-q4_K_M"
            elenchus_model = "mistral:7b-instruct-v0.3-q3_K_S"
            aporia_model = "llama3.1:8b-instruct-q2_K"
            maieutics_model = "llama3.1:8b-instruct-q2_K"
            dialectic_model = "phi4-mini:3.8b-q4_K_M"
        else:
            arbiter_model = elenchus_model = aporia_model = maieutics_model = dialectic_model = "llama3.1:8b-instruct-q2_K"

        # The Orchestrator: Lowest temperature (0.0) for high precision, logical routing, and intent classification.
        self.arbiter_llm = ChatOllama(model=arbiter_model, temperature=0.0, backend=ollama_backend)

        # The Adversary: Tuned for logical rigor and cross-examination. Low temperature (0.1) for solid, factual responses
        self.elenchus_llm = ChatOllama(model=elenchus_model, temperature=0.1, backend=ollama_backend)

        # The Puzzler: Slightly higher temperature (0.7) to allow for creative analogies and the generation of 'productive doubt' paradoxes.
        self.aporia_llm = ChatOllama(model=aporia_model, temperature=0.7, backend=ollama_backend)

        #The Midwife: Balanced temperature (0.5) for clear, constructive scaffolding and analogy-based knowledge synthesis.
        self.maieutics_llm = ChatOllama(model=maieutics_model, temperature=0.5, backend=ollama_backend)

        #The Auditor: Evaluates state transitions and verifies concept mastery. Lowest temperature (0.0) for objectiveness and determinism.
        self.dialectic_llm = ChatOllama(model=dialectic_model, temperature=0.0, backend=ollama_backend)

        # The system "objective" prompts
        self.prompts = {
            "arbiter": "Analyze the conversation and decide which agent should respond next. Reply with exactly one token: elenchus, aporia, or maieutics.",
            "elenchus": "Find logical contradictions in the user's statement. Be sharp and persistent.",
            "aporia": "Create a sense of wonder and doubt. Use paradoxes to show why this topic is difficult.",
            "maieutics": "Use analogies to help the user discover the truth themselves. Do not give the answer.",
            "dialectic": "Evaluate the user's progress. Assign a single mastery score value between 0.0 to 1.0."
        }

    def _parse_score(self, ai_output: str) -> float:
        """
        Extract a numerical mastery score from the Dialectic agent's output string.
        Returns:
            float: The parsed score, or 0.0 if not found.
        """
        """Helper to extract a numerical score from the Dialectic agent's text."""
        import re
        try:
            # Look for any decimal number in the response (e.g., '0.8' or 'Score: 0.5')
            match = re.search(r"(\d\.\d)", ai_output)
            if match:
                return float(match.group(1))
            return 0.0 # Default if no score found
        except Exception:
            return 0.0

    def _parse_next_agent(self, ai_output: str) -> str:
        """
        Extract a canonical next-agent name from the Arbiter's output.
        Searches for known agent names in the model output and returns the first match.
        Falls back to 'maieutics' when no known agent is found.
        """
        if not ai_output:
            return "maieutics"
        text = ai_output.lower()
        candidates = ["arbiter", "elenchus", "aporia", "maieutics", "dialectic"]
        for c in candidates:
            if c in text:
                return c

    def arbiter_node(self, state: SocraticState):
        """
        Arbiter node: Decides which agent should handle the next step based on the conversation state.
        Returns:
            dict: Contains the name of the next agent.
        """
        # get prompt from prompt dict
        prompt = self.prompts["arbiter"]
        
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = self.arbiter_llm.invoke(messages)
        
        # We extract the name of the next agent (e.g., 'elenchus') 
        # so the graph knows which edge to take.
        next_agent = self._parse_next_agent(response.content)
        return {"next_agent": next_agent}

    def elenchus_node(self, state: SocraticState):
        """
        Elenchus node: Challenges the user's statements for logical rigor and contradiction.
        Returns:
            dict: Contains the agent's response messages.
        """
        prompt = self.prompts["elenchus"]

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = self.elenchus_llm.invoke(messages)

        return {"messages": [response]}

    def aporia_node(self, state: SocraticState):
        """
        Aporia node: Introduces productive doubt and paradoxes to deepen understanding.
        Returns:
            dict: Contains the agent's response messages.
        """
        prompt = self.prompts["aporia"]

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = self.aporia_llm.invoke(messages)

        return {"messages": [response]}

    def maieutics_node(self, state: SocraticState):
        """
        Maieutics node: Guides the user to discover knowledge through analogies and scaffolding.
        Returns:
            dict: Contains the agent's response messages.
        """
        prompt = self.prompts["maieutics"]

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = self.maieutics_llm.invoke(messages)

        return {"messages": [response]}

    def dialectic_node(self, state: SocraticState):
        """
        Dialectic node: Evaluates the user's mastery and assigns a score.
        Returns:
            dict: Contains the mastery score.
        """
        prompt = self.prompts["dialectic"]

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = self.dialectic_llm.invoke(messages)
        
        score = self._parse_score(response.content) 
        return {"mastery_score": score}
