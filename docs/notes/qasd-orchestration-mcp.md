# Note (on hold): Qasd behind an orchestrator, tools over MCP

Parked for later. The flow: Qasd routes each message (direct_reply, identity, retrieve, function_call; low
confidence goes to the planner), an orchestrator in the application picks the tools, a planner chat model calls them.

## Already in the library
- Planner: `ChatGenerator` (any model the library loads; tool calls parsed from the model's own chat template).
- Tool loop and history: `Conversation` (tool rounds, results sent back, round limit).
- Tools: `ToolRegistry` (from functions / methods / `[Tool]`), with per-tool approval (the confirmation before an
  action), allow rules and timeouts.
- Retrieve tools: `RetrievalIndex`, `VectorIndex`, `Bm25Index`, `TextEncoder`, `CrossEncoder`, and
  `RetrievalTools.Search` (a ready search tool over an index).
- MCP: `McpTools.ConnectStdioAsync` / `ConnectHttpAsync` (use any MCP server's tools) and `McpTools.ServerTools`
  (serve a registry's tools, its approvals and rules applying).
- Serving: `NeuralSharp.AspNetCore` (chat endpoints, Ollama-compatible API with tools), `ModelHost`.

## Missing
- The orchestrator (intent → tool set, the confidence rule, "yes after my question = confirm"): application code.
- Structured conversation state (doctor, day, phone): `Conversation` keeps the history only.
- Qasd reading the bot's previous question (settles "نعم", "Thursday", phone numbers): request format and data
  (app); `AnswerScorer` already takes whole conversations.
- A runnable MCP server sample and an end-to-end sample (Qasd → planner → retrieve tool + approval-gated booking).

## Risk to measure first
How well a small model (Qwen2.5-0.5B) plans with tools: choosing the tool and filling arguments from earlier turns.
Fixes: tune the planner with nstune on conversations with tool calls, or a larger planner model only for the
messages Qasd sends it.
