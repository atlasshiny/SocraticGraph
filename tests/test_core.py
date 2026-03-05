"""
Unit tests for core logic that does NOT require Ollama or any external service.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent_graph import _route_after_dialectic
from agents import SocraticAgents
from history import (
    load_history,
    reset_history,
    save_history,
)

class TestParseScore(unittest.TestCase):
    """Tests for SocraticAgents._parse_score (no LLM needed)."""

    @classmethod
    def setUpClass(cls):
        # Instantiation needs ChatOllama; bypass by building the object without __init__
        cls.agents = object.__new__(SocraticAgents)

    def test_clean_decimal(self):
        self.assertAlmostEqual(self.agents._parse_score("0.85"), 0.85)

    def test_score_in_sentence(self):
        self.assertAlmostEqual(
            self.agents._parse_score("The mastery score is 0.72 based on analysis."),
            0.72,
        )

    def test_perfect_score(self):
        self.assertAlmostEqual(self.agents._parse_score("1.0"), 1.0)

    def test_zero_score(self):
        self.assertAlmostEqual(self.agents._parse_score("0.0"), 0.0)

    def test_no_number_returns_zero(self):
        self.assertAlmostEqual(self.agents._parse_score("no score here"), 0.0)

    def test_empty_string(self):
        self.assertAlmostEqual(self.agents._parse_score(""), 0.0)

class TestParseNextAgent(unittest.TestCase):
    """Tests for SocraticAgents._parse_next_agent."""

    @classmethod
    def setUpClass(cls):
        cls.agents = object.__new__(SocraticAgents)

    def test_exact_elenchus(self):
        self.assertEqual(self.agents._parse_next_agent("elenchus"), "elenchus")

    def test_exact_aporia(self):
        self.assertEqual(self.agents._parse_next_agent("aporia"), "aporia")

    def test_exact_maieutics(self):
        self.assertEqual(self.agents._parse_next_agent("maieutics"), "maieutics")

    def test_uppercase_ignored(self):
        self.assertEqual(self.agents._parse_next_agent("ELENCHUS"), "elenchus")

    def test_agent_in_sentence(self):
        self.assertEqual(
            self.agents._parse_next_agent("I think elenchus is best here."),
            "elenchus",
        )

    def test_multiline_with_agent_on_own_line(self):
        text = "Some preamble\naporia\nMore text"
        self.assertEqual(self.agents._parse_next_agent(text), "aporia")

    def test_empty_falls_back(self):
        self.assertEqual(self.agents._parse_next_agent(""), "maieutics")

    def test_none_falls_back(self):
        self.assertEqual(self.agents._parse_next_agent(None), "maieutics")

class TestRouteAfterDialectic(unittest.TestCase):
    """Tests for _route_after_dialectic conditional edge."""

    def test_mastery_reached_ends(self):
        state = {"mastery_reached": True}
        from langgraph.graph import END
        self.assertEqual(_route_after_dialectic(state), END)

    def test_mastery_not_reached_continues(self):
        state = {"mastery_reached": False}
        self.assertEqual(_route_after_dialectic(state), "arbiter")

    def test_missing_key_continues(self):
        self.assertEqual(_route_after_dialectic({}), "arbiter")

if __name__ == "__main__":
    unittest.main()
