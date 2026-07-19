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

## Execution Coordination

`ExecutionCoordinator` integrates the previous pieces without replacing them:

1. `ExecutionDecisionEngine` classifies the original request.
2. `ToolProposalBuilder` or `ToolChainProposalBuilder` builds a structured
   proposal.
3. Complete proposals are converted through their safe conversion methods.
4. `SingleToolRunner` or `ToolChainRunner` executes.
5. Confirmation responses are delegated back to the runner that owns the
   pending `confirmation_id`.

The coordinator returns `ExecutionCoordinationResult` with a uniform status,
mode, decision, optional proposal, optional runner result, message, missing or
ambiguous information, validation errors, confirmation id and executed flag.

Coordinator statuses are:

- `DIRECT_RESPONSE_REQUIRED`
- `INFORMATION_REQUIRED`
- `AMBIGUOUS_REQUEST`
- `UNSUPPORTED`
- `VALIDATION_FAILED`
- `CONFIRMATION_REQUIRED`
- `EXECUTED`
- `CANCELLED`
- `FAILED`

The coordinator does not generate LLM answers, parse new arguments, call
`ToolExecutor`, bypass selectors or validators, add artificial confirmations,
retry, rollback, run a planner, or modify the conversational orchestrator. The
CLI `python -B -m tools.run_execution_request "<prompt>"` is an isolated manual
entry point for this phase.

## Text Conversation Integration

`AtlasOrchestrator.process_prompt()` now sends text turns through
`ExecutionConversationController` before the previous conversational flow. The
controller calls `ExecutionCoordinator.execute(prompt)` when no confirmation is
pending. If the result is `DIRECT_RESPONSE_REQUIRED`, control returns to the
existing planner, router, memory and agent flow; Atlas does not invent a fixed
direct response.

The controller owns one session-level pending confirmation id. While it is
present, the next text input is treated as the confirmation response and is sent
to `ExecutionCoordinator.confirm(id, response)`. Ambiguous replies keep the id
pending and ask again. Accepted or rejected replies clear the id, so a completed
confirmation cannot be reused and a second pending confirmation is not created
accidentally.

`ExecutionResultPresenter` formats coordination statuses for the user:

- `EXECUTED`: shows the real tool or chain output without dumping dataclasses.
- `INFORMATION_REQUIRED`: asks for the missing fields.
- `AMBIGUOUS_REQUEST`: asks the user to clarify ambiguous fields.
- `UNSUPPORTED`: reports that Atlas does not have the required tool.
- `VALIDATION_FAILED`: gives a readable validation failure.
- `CONFIRMATION_REQUIRED`: describes the pending action and asks whether to
  continue, without exposing the confirmation id.
- `CANCELLED`: reports cancellation.
- `FAILED`: reports a uniform failure.

Text integration remains out of scope for voice, wake word, autonomous memory,
multi-agent flows, ReAct, advanced planning, LLM-generated proposals, retries,
parallelism and rollback.

## Multi-Turn Clarification

When the coordinator returns `INFORMATION_REQUIRED` or `AMBIGUOUS_REQUEST`, the
text controller stores a single `PendingClarification` instead of executing:

- `original_text`
- `mode`
- `proposal`
- `missing_information`
- `ambiguous_information`
- `requested_fields`

The next text input is handled as a clarification answer before it can become a
new request. The resolver fills only requested fields, rebuilds a complete text
request, and sends it back through `ExecutionCoordinator.execute()`. This keeps
schema validation, proposal conversion, confirmation and execution in the
existing layers.

Partial answers are kept as an updated pending clarification and Atlas asks only
for the remaining fields. Empty or irrelevant answers do not execute tools and
do not clear the pending operation. The supported clarification context is
limited to simple answers such as paths, directory names, write content, target
window names, or a full replacement command for the pending chain.

Cancellation words for clarification are `cancelar`, `cancela`, `cancelalo`,
`cancelala`, `olvidalo`, `olvídalo`, and `salir de esta operacion`. Cancelling
clears the pending clarification and does not execute anything.

If a clearly separate new order arrives while a clarification is pending, Atlas
does not merge it with the previous operation. The current policy is to keep the
pending clarification and ask the user to answer it or cancel it first.

Clarification and confirmation are mutually exclusive. Once clarification
produces a `CONFIRMATION_REQUIRED` result, the clarification state is cleared and
only the confirmation id remains pending. Completion, cancellation and failures
also clear the clarification state.

## Pending Confirmation Modification

During a pending confirmation the text controller stores an explicit
`PendingConfirmationContext`:

- `confirmation_id`
- `mode`
- `owner` (`single` or `chain`)
- `original_text`
- `proposal`
- `execution_result`
- `action_summary`

The context is conversational state only. Execution still belongs to
`SingleToolRunner` and `ToolChainRunner`; the controller never calls
`ToolExecutor` directly.

Before any pending operation can execute, the next input is classified as:

- `CONFIRM`: delegate to `ExecutionCoordinator.confirm(id, response)`.
- `REJECT` or `CANCEL`: cancel through the owning runner and clear state.
- `AMBIGUOUS`: execute nothing and keep the same confirmation pending.
- `INSPECT`: describe the tool or chain, affected path, reasonable content
  summary, completed chain steps and pending step without exposing the id.
- `MODIFY`: apply only explicitly provided argument changes, revalidate through
  the existing selector and `ArgumentValidator`, invalidate the old id and issue
  a new confirmation.
- `REPLACE`: only when the user clearly cancels/replaces the old operation;
  cancel the old confirmation and process the remaining text as a new request.

Single-tool modification reissues the pending `ValidatedToolRequest` with merged
arguments for the same intent. The previous `confirmation_id` is removed before
the new confirmation is stored, so stale ids return `confirmation_not_found` at
runner level or `FAILED` at coordinator level and never execute.

Chain modification is limited to the currently pending step. Completed step
results are preserved and are not repeated; references such as
`${steps.read.output.content}` remain in the chain definition while the
underlying pending single-tool request holds the already resolved content needed
for confirmation. A modified pending chain step receives a fresh
`confirmation_id`.

Already executed chain steps are immutable in this phase. For example, after
`read` has executed and the chain is paused before `write`, a request to read a
different source is rejected with an explanation that the user must cancel and
start a new operation. Atlas does not silently rewrite completed history.

A new command without an explicit cancel/replace signal is not treated as a
replacement. Atlas asks the user to confirm, modify, inspect, or cancel the
pending operation first. This prevents mixing arguments from unrelated
operations.

Natural confirmation prompts describe the pending action instead of showing only
an internal state. Long content is summarized by size; the full internal value is
kept for execution.

This phase remains deterministic and does not add an LLM for modification
parsing, general conversational memory, voice, wake word, planner behavior,
ReAct, retries, rollback, parallelism, multiple pending operations, or
retroactive mutation of executed chain steps.
