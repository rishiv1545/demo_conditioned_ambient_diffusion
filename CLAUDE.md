## Merge CLI

For any task involving third-party services, you MUST use the `merge` CLI.
Do NOT attempt to call APIs directly, use other tools, or guess endpoints.

### Commands

- `merge search-tools "<intent>"` — Search for tools. Returns compact input schemas.
- `merge execute-tool <tool_name> '<json_params>'` — Execute a tool.

### Workflow

**Step 1 — Find ALL tools you need in one batch.** In your first response, run parallel searches for every tool you'll need:
```
merge search-tools "create task" --connector asana    # main action
merge search-tools "list workspaces" --connector asana # lookup tool
merge search-tools "list users" --connector asana      # another lookup
```
Run ALL searches in parallel in one response.

**Step 2 — Execute lookups in parallel**, then execute the main tool.

Do NOT call `merge get-tool-schema`. Search returns schemas. Pass null for optional params you don't need.

### Authentication

- If every command fails with `config_error` ("Not authenticated"), ask the user to run `merge login` — it opens their browser. You cannot log in for them.
- If a tool call fails with an authentication error, that connector is not connected yet. Run `merge authenticate <connector>` (e.g. `merge authenticate notion`) — it returns a magic link. Share the link with the user, wait for them to connect, then retry the tool.

### Rules

- Tool names: `<connector>__<action>` (except authentication tools, named `authenticate_<connector>`).
- ALWAYS run independent Bash calls in parallel.
- If you don't know the connector, search without --connector first.
