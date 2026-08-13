# Multi-Step Legal AI Agent with LangGraph

This repository expands our AI Agent architecture from single function-calling into a **multi-step, state-driven workflow using LangGraph**.

## Key Upgrades in this Release

1. **Explicit State Management (`AgentState`)**:
   - Uses `Annotated[list[BaseMessage], add_messages]` to handle conversation history seamlessly.
   - Tracks execution meta-state like `current_step` and `error_count`.

2. **Sequential Multi-Tool Execution**:
   - `legal_code_search`: Retrieves exact penal/statutory references.
   - `plain_english_explainer`: Converts raw statutory text into simplified explanations.

3. **Conditional Routing & Fallback**:
   - Implements dynamic routing via `should_continue`.
   - Intercepts tool execution errors (`TOOL_ERROR`) and routes to an `error_recovery` node to prevent flow crashes.

4. **LangSmith Integration & Observability**:
   - Full execution tracing enabled for step-by-step state inspection.

---

## 📊 Graph Architecture
