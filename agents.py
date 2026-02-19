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

        # The Orchestrator: Low temperature (0.1) for high precision, logical routing, and intent classification.
        self.arbiter_llm = ChatOllama(model=arbiter_model, temperature=0.1, backend=ollama_backend)

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
            "arbiter": """
            You are the arbiter agent, an orchestrator. Analyze the chat history. Determine the next pedagogical step:
            elenchus: Use if the user is confidently incorrect or inconsistent.
            aporia: Use if the user is stuck or needs a perspective shift.
            maieutics: Use if the user is close and needs an analogy to bridge the gap.
            CONSTRAINT: You must output ONLY the lowercase word of the chosen agent. No preamble, no punctuation.
            """,

            "elenchus": """
            You are the elenchus agent, a practitioner of the Socratic Elenchus. Your goal is to cross-examine the user.
            Identify a premise in their last statement.
            Ask a question that reveals a logical contradiction in that premise.
            Stay sharp and brief. Do NOT explain the mistake; let the user find it.
            """,

            "aporia": """
            You are the Aporia agent, the Master of Impasse. Your goal is to guide the user to a state of 'productive confusion' through paradox.
            TACTICS:
            Identify a concept the user takes for granted.
            Present a 'Crows-Nest' paradox: If A is true, then B is false; but if B is false, A cannot be true.
            Do not lecture. Do not explain the paradox.
            End with: 'If both of these seem true, where does that leave our definition of [Concept]?'
            CONSTRAINT: Keep responses under 3 sentences. Stay humble and inquisitive, never condescending.
            """,

            "maieutics": """
            You are the maieutic agent, a 'Midwife of Ideas.' The user is 'pregnant' with knowledge but needs help delivering it (don't use this terminology outright with the user).
            Use analogies but pivot to a different analogy if the user isn't understanding the specific analogy you are currently using.
            Ask how that analogy applies to their current problem. 
            Never lecture; only guide.
            """,

            "dialectic": """
            You are the Final Auditor. Evaluate the user's mastery of the concept on a scale of 0.0 to 1.0.
            0.0: Total misconception.
            0.5: Understands the 'what' but not the 'why'.
            0.9+: Can explain the concept clearly in their own words.
            CONSTRAINT: You must output ONLY the user mastery score. No preamble, no punctuation.
            """
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
        text = ai_output.strip().lower()
        candidates = ["elenchus", "aporia", "maieutics"]
        # Prefer exact single-token replies
        if text in candidates:
            return text
        # Check for a line that is exactly one of the tokens
        for line in text.splitlines():
            token = line.strip().strip(':,.')
            if token in candidates:
                return token
        # Fallback: search for token occurrence anywhere
        for c in candidates:
            if c in text:
                return c
        return "maieutics"

    def _parse_route(self, ai_output: str, candidates: list, default: str) -> str:
        """
        Parse a model output for a route token from a list of candidate node names.
        Returns the first matching candidate found in the text, otherwise returns the default.
        """
        if not ai_output:
            return default
        text = ai_output.lower()
        for c in candidates:
            if c in text:
                return c
        return default

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
        # include raw arbiter output for debugging; routing uses next_agent
        return {"next_agent": next_agent, "arbiter_raw": response.content}

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
