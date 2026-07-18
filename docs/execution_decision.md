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
- Proposal converts one `SINGLE_TOOL` decision and the original text into a
  `StructuredToolProposal`: candidate intent, extracted arguments, missing or
  ambiguous fields, validation errors, confidence, reason, status and original
  source text.
- `ToolIntent` is the executable structured request shape consumed later by
  selection, validation and runners. Only a `COMPLETE` proposal can become a
  `ToolIntent`; incomplete, ambiguous and unsupported proposals stay inert.
- Selection maps a structured `ToolIntent` to one registered tool.
- Execution validates arguments and runs tools through the existing runners.

## Proposal States

- `COMPLETE`: extracted arguments pass the registered schema and validator.
- `INCOMPLETE`: required arguments are missing; no values are invented.
- `AMBIGUOUS`: at least one value is too vague to use safely.
- `UNSUPPORTED`: the decision is not `SINGLE_TOOL`, the candidate is not
  registered, or validation fails for a reason other than missing arguments.

## Tool Proposal Limits

`ToolProposalBuilder` does not execute tools, ask for confirmation, call the
`SingleToolRunner`, build chains, use memory, use voice, call agents, call an
LLM, or write files. The initial extraction is deterministic and schema-aware;
the interface is kept separate so a later LLM-backed mapper can preserve the
same proposal contract.

## Chain Proposals

`ToolChainProposalBuilder` handles only `ExecutionDecision(mode=TOOL_CHAIN)`.
It decomposes the original request into ordered, linear
`StructuredToolChainStepProposal` entries and reuses `ToolProposalBuilder` for
each single-tool step where possible.

A chain proposal is different from an individual proposal:

- An individual proposal describes one future `ToolIntent`.
- A chain proposal describes ordered future `ToolChainStep` definitions,
  dependencies between steps, and reference strings in arguments.

Chain proposal states are `COMPLETE`, `INCOMPLETE`, `AMBIGUOUS`, and
`UNSUPPORTED`. Any incomplete, ambiguous, unsupported, duplicated, future, or
unknown step reference prevents conversion to a `ToolChainStep` tuple.

Dependencies are represented with `depends_on` step ids and references use only
the syntax already understood by `ToolChainRunner`:

- `${steps.<id>.output}`
- `${steps.<id>.output.<field>}`
- `${steps.<id>.output.0}`

This phase validates candidate tools, step ids, raw arguments, schemas and
reference direction/existence. `ToolChainRunner` remains the only component that
resolves references against real tool results, asks confirmations through the
single-tool runner, and executes tools.
