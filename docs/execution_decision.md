# Execution Decision

## Responsibilities

`ExecutionDecisionEngine` classifies a user request into one of three modes:

- `DIRECT_RESPONSE`: no registered tool capability is needed.
- `SINGLE_TOOL`: one registered tool intent appears to be enough.
- `TOOL_CHAIN`: multiple registered tool intents appear necessary in sequence.

The classifier returns an `ExecutionDecision` with a short technical reason,
normalized confidence, candidate registered tool intents, and required
capabilities.

## Phase Limits

This phase does not execute tools, build full arguments, create final chains,
ask for confirmations, call an LLM, use voice, or involve agents.

The initial implementation is deterministic and rule-based. The contract is
kept stable so a later model-backed classifier can be connected without changing
callers.

## Decision vs Selection vs Execution

- Decision says which processing mode is likely needed.
- Selection maps a structured `ToolIntent` to one registered tool.
- Execution validates arguments and runs tools through the existing runners.
