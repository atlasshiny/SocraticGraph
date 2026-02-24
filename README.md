# SocraticGraph

Lightweight orchestration for a Socratic-style multi-agent loop using LangGraph and Ollama-backed LLMs.

## Overview

- Purpose: run a small ensemble of role-based agents (Arbiter, Elenchus, Aporia, Maieutics, Dialectic) to guide a user through Socratic dialogue and evaluate mastery.
- Flow: `arbiter` -> chosen agent (`elenchus` | `aporia` | `maieutics`) -> `dialectic` -> END (user provides next input).

This repository provides a minimal CLI runner, agent implementations, and a graph that routes between agents.

## Repo Structure

- `main.py` — CLI entrypoint, reads user input and streams the agent graph.
- `agents.py` — implementations of the role-based nodes and prompts.
- `agent_graph.py` — constructs the `StateGraph` and edges between nodes.
- `agent_state.py` — typed state shape used by the graph.
- `history.py` — keeps track of message history and recovering previous message history.
- `requirements.txt` — Python dependencies (install into a venv).

## Quickstart

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Ensure Ollama is installed and configured on your machine to use `langchain_ollama` as the backend.

4. Run the CLI:

   ```powershell
   python main.py
   ```

Type a question when prompted. Enter `options` to change settings. Enter `quit` or `exit` to terminate.

## Configuration and Notes

- Models and temperatures are set in `agents.py`. Change model names there to swap models or reduce VRAM usage (choose smaller models for limited GPUs).
- The `arbiter` prompt is intentionally strict: it is instructed to reply with a single token (one of `elenchus`, `aporia`, `maieutics`) to avoid routing ambiguity.
- The graph is defined in `agent_graph.py`. The current flow ensures evaluation by `dialectic` after each chosen-agent response and then ends the run so the user must enter the next prompt to continue.

## Debugging

- If the CLI appears blocked after output:
  - Confirm the graph returns to `END`. The stream should complete and return control to the prompt.
  - Enable/inspect raw arbiter outputs (the code includes `arbiter_raw` in arbiter node output) to see what the arbiter returned.

- GPU usage:
  - Monitor with `nvidia-smi` (Windows: run in a separate terminal).
  - Ollama controls model device usage; confirm the Ollama install and models support GPU.

- Mastery score parsing:
  - The `dialectic` node uses a parser to extract a numeric mastery score. If you see out-of-range values, tighten the `dialectic` prompt (request a single numeric token) and/or improve the parser in `agents.py`.

## Development

- To add or modify nodes: update `agents.py` with a new node function and add a node/edge in `agent_graph.py`.
- The project uses `langgraph` to build and run the directed state graph — consult that library's docs for advanced routing and conditional-edge features.

## Contributing

Open an issue or submit a PR with improvements. For prompt engineering changes, prefer small, testable prompt edits and include example inputs.

## License

This project is released under the MIT License. You can find a copy of the license in the `LICENSE` file (create one if it does not exist).

Short summary: you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to including the original MIT license and copyright notice.
