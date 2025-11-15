Generate a task directory containing the core files described below. The typical files are:

- `task.yaml` — instruction + metadata the agent follows.
- `run-tests.sh` — the script that installs dependencies and runs the test harness.
- `Dockerfile` — the container image used to create the sandbox for the agent.
- `docker-compose.yaml` — helper compose file used to run the task containers.
- `tests/` — directory containing test files parsed by the configured test parser (e.g., pytest tests).
- `runs/` — (not inside a task skeleton) recorded run outputs / logs from previous runs.

## File reference and how to modify each

### `task.yaml` — task instruction and metadata

Purpose
- Holds the instruction the agent should follow, author metadata, category, tags, parser selection, and timeouts.

Key fields and examples
- `instruction` (string, multi-line): human-readable task description. This is what the agent will try to accomplish.
- `author_name`, `author_email`: contact info for the author.
- `difficulty`, `category`, `tags`: categorization for filtering/search.
- `parser_name`: which parser to use to interpret test output (e.g., `pytest`).
- `max_agent_timeout_sec`, `max_test_timeout_sec`: timeouts for agent runtime and individual tests.
- `run_tests_in_same_shell`: boolean controlling whether tests run inside the same shell session as the agent steps.
- `estimated_duration_sec`, `expert_time_estimate_min`, `junior_time_estimate_min`: guidance values for runners.

How to modify
- Change `instruction` to clarify the goal — prefer explicit, minimal steps and expected artifacts (paths, filenames, exact file contents when checking file-creation tasks).
- Increase `max_agent_timeout_sec` for longer-running tasks (compilation, training, network downloads). Keep limits reasonable to avoid runaway containers.
- Switch `parser_name` to a different parser if your tests produce other output formats. Ensure the corresponding parser is available in the test harness.
- Add tags and categories to help with discoverability.

Best practices
- Keep the instruction deterministic (avoid asking agents to fetch network resources unless the image provides that access).
- Include exact file paths the agent should produce where feasible — e.g., `/app/hello.txt` — since tests usually check exact paths.

### `run-tests.sh` — install & run test harness

Purpose
- A shell script executed inside the task container to install test dependencies and run the tests. The default script uses `uv` to manage a Python test environment and runs `pytest`.

Typical actions inside this script
- Install system utilities (curl, etc.).
- Install `uv` and Python dependencies (via `uv add`), then run `uv run pytest` on the test module.

How to modify
- If your task uses Node, Ruby, or other runtimes, replace the `uv`/`pytest` calls with the corresponding installer and test runner (e.g., `npm install && npm test`).
- Add additional `apt-get` packages if the tests require system-level dependencies (e.g., compilers, libraries).
- Ensure the script exits non-zero on failures so the test harness reports a failed run.
- If tests must run in the same shell session as the agent (to inspect state changes), set `run_tests_in_same_shell: true` in `task.yaml` and adjust the script accordingly.

Example modifications
- To run a Node test suite, replace the `uv` section with `apt-get install -y nodejs npm` and run `npm ci && npm test`.

### `Dockerfile` — sandbox environment

Purpose
- Defines the container image that will be used to run the agent and tests. The default base images are published by the terminal-bench project and include helpful tooling.

How to modify
- Use a different base image (for example, `python:3.11-slim`) if you need a specific runtime or package set.
- Set `WORKDIR /app` and copy any pre-seeded files into the image with `COPY` if the task needs scaffolding.
- Add `RUN apt-get update && apt-get install -y <packages>` lines for system deps.
- Avoid installing heavy or time-consuming packages at runtime — prefer small, purpose-built base images where possible.

Notes
- The default images include `tmux` and `asciinema` for interactive debugging. If you need debugging, keep these installed.

### `docker-compose.yaml` — container wiring

Purpose
- Compose file used to start the task container and map volumes for logs and test artifacts. Usually you don't need to change this, but it can be extended to add supporting services.

How to modify
- Add services if the task requires a database or external service (e.g., Redis, PostgreSQL). Make sure to document required environment variables.
- Modify `volumes` if you want custom log paths or to expose host files into the container.

Caution
- Keep sensitive secrets out of the compose file. Use environment variables provided by your CI or secure vault instead.

### `tests/` — test harness files

Purpose
- Contains the actual test code or scripts that validate the agent's output. For Python tasks, this typically includes `test_outputs.py` or similar pytest modules.

How to modify
- Add assert-based tests that check for expected artifacts, file contents, file permissions, and exit codes.
- Use the parser the task declares (e.g., pytest). Tests should print or return outputs the parser understands.
- Keep tests deterministic: avoid timing-sensitive assertions unless strictly necessary.

Example: file-creation test
- A test that asserts `/app/hello.txt` exists and contains the exact string `Hello, world!`.

### `runs/` — previous run artifacts

Purpose
- `runs/` typically contains previous run directories and logs (timestamps or run IDs). Inspect these for debugging failing tasks.

How to use
- Open the latest run folder and inspect stdout/stderr logs, container logs, and any saved `asciinema` recordings.
- If you need to reproduce locally, use the `docker-compose.yaml` and `Dockerfile` to start the same image and run the agent commands.

## Common edits & examples

- Changing the required output file: update `instruction` in `task.yaml` and update or add tests in `tests/` to assert the new path and contents.
- Adding dependencies: add `apt-get` installs to the `Dockerfile` and add package installs to `run-tests.sh` (or pre-bake them into the image).
- Using another language: replace the `uv`/`pytest` calls in `run-tests.sh` with the language-specific test commands and update `parser_name` if needed.

## Checklist before publishing a task

- Instruction is explicit and deterministic.
- Tests are deterministic and fail loudly (non-zero exit code) on problems.
- Timeouts are adequate for worst-case runs.
- Image contains everything required or `run-tests.sh` installs dependencies reproducibly.
- No secrets checked into `Dockerfile` or `docker-compose.yaml`.

## Contract & edge cases

Contract (what the task should guarantee)
- Input: a clean container created from `Dockerfile`.
- Output: tests in `tests/` pass (test harness returns success) and any files indicated in `instruction` exist with the expected content.

Edge cases to consider
- Empty or missing working directory: `run-tests.sh` should check `PWD` and fail fast with a helpful error.
- Long-running installs: prefer pre-built images to avoid flakiness.
- Parser mismatch: ensure `parser_name` matches the test runner output format.