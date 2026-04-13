# PRD: End-to-End Documentation & Testing for `adm` CLI

## Introduction

The `adm` CLI has grown to 7 commands (`init`, `new`, `run`, `trigger`, `clean`, `document`, `add dqx`) but the README only documents 5. Recent additions (`document`, `trigger`) plus the new gitignore/low-signal filtering have no user-facing docs. Additionally, there is no end-to-end validation covering the full lifecycle — from `adm document` against various sources to the MLOps pipeline running successfully in Databricks, to notebook-native usage. This PRD covers a full README audit, real end-to-end testing against Databricks, and notebook compatibility hardening.

## Goals

- Document every `adm` command in the README with consistent style
- Validate `adm document` end-to-end against local projects and remote Git URLs
- Validate the full MLOps pipeline (`init` -> `run` -> job completes) in Databricks
- Ensure the package works when pip-installed inside a Databricks notebook
- Add automated tests where feasible; manual verification with real endpoints where not

## User Stories

### US-001: README — Document `adm document` command
**Description:** As a user reading the README, I want to see the `document` command listed with usage examples so I know how to generate a repository review.

**Acceptance Criteria:**
- [ ] `adm document` section added to Commands in README, matching existing style (minimal: signature, description, example)
- [ ] Mentions smart filtering: respects `.gitignore`, skips low-signal artifacts (mlruns/, logs, binary files)
- [ ] Shows example for local path, remote Git URL, and default (current directory)
- [ ] Mentions endpoint auto-selection behavior in one sentence
- [ ] Lists key options: `--source`, `--output`, `--endpoint`, `--max-file-chars`, `--max-total-chars`

### US-002: README — Document `adm trigger` command
**Description:** As a user, I want to see `adm trigger` documented so I know it exists as a lightweight alternative to `adm run`.

**Acceptance Criteria:**
- [ ] `adm trigger` section added to Commands in README
- [ ] Clearly states it re-runs an already-deployed job via SDK (no CLI needed, no redeploy)
- [ ] Mentions notebook-native usage: `from az_databricks_mlops import run_training_job`
- [ ] Shows the `--target` option

### US-003: README — Document programmatic API for notebooks
**Description:** As a user working inside a Databricks notebook, I want the README to show me how to pip-install the package and call its functions directly.

**Acceptance Criteria:**
- [ ] New "Notebook Usage" section added to README
- [ ] Shows `pip install` command (from GitHub)
- [ ] Shows `from az_databricks_mlops import run_training_job` example with argument
- [ ] Shows `from az_databricks_mlops import review_repository` example for document generation
- [ ] Notes which commands work in notebooks (trigger, document) vs. which require local CLI (init, run)

### US-004: README — Full audit for staleness
**Description:** As a maintainer, I want the README to be accurate against the current CLI surface.

**Acceptance Criteria:**
- [ ] All 7 commands listed in Commands section
- [ ] "The complete flow" diagram updated if `trigger` fits in there
- [ ] Package version / naming references are accurate (`az-databricks-mlops`, `adm`)
- [ ] No dead references to removed features or wrong flag names

### US-005: E2E test — `adm document` on local `examples/messy-ml-project`
**Description:** As a developer, I want to verify that `adm document` generates a valid review document when pointed at the bundled example project.

**Acceptance Criteria:**
- [ ] Run `adm document --source examples/messy-ml-project --output /tmp/review-messy.md --endpoint <available-endpoint>`
- [ ] Command exits 0
- [ ] Output file exists and contains a Markdown review with headings
- [ ] Output file contains the HTML comment header (source, endpoint, timestamp)
- [ ] `.adm_internal/latest-review-prompt.txt` was written
- [ ] Omitted files include expected artifacts (`.pkl`, `mlruns/`, etc.)

### US-006: E2E test — `adm document` on a remote Git URL
**Description:** As a developer, I want to verify that `adm document` can clone and review a remote repository.

**Acceptance Criteria:**
- [ ] Pick a suitable Git URL — either a 10x sibling project (e.g. `brand-assistant` via its SSH remote) or a small public ML repo
- [ ] Run `adm document --source <git-url> --output /tmp/review-remote.md --endpoint <available-endpoint>`
- [ ] Command exits 0 and produces a valid Markdown review
- [ ] Temp clone directory is cleaned up after completion
- [ ] If the remote repo has a `.gitignore`, its patterns are respected in the snapshot

### US-007: E2E test — Full MLOps pipeline (`init` -> `run` -> job success)
**Description:** As a developer, I want to verify the core MLOps flow works end-to-end in Databricks.

**Acceptance Criteria:**
- [ ] Run `adm clean` in `examples/messy-ml-project` to start fresh
- [ ] Run `adm init` with appropriate flags (non-interactive)
- [ ] Run `adm run` (deploys bundle, starts training job)
- [ ] Wait for the Databricks job to complete (check via SDK or job URL)
- [ ] Job finishes successfully — model registered in Unity Catalog
- [ ] Run `adm trigger` to re-trigger the same job (verify SDK path works too)

### US-008: E2E test — `adm document` from Databricks notebook
**Description:** As a user pip-installing inside a Databricks notebook, I want `review_repository()` to work when pointed at a local repo path cloned into the workspace.

**Acceptance Criteria:**
- [ ] Create a test notebook that: `%pip install git+<repo-url>`
- [ ] Notebook imports `from az_databricks_mlops import review_repository`
- [ ] Clones a small repo into `/tmp/` via `!git clone --depth 1 <url> /tmp/test-repo`
- [ ] Calls `review_repository(source="/tmp/test-repo", output_path="/tmp/review-notebook.md")`
- [ ] Review document is generated successfully
- [ ] Verify no subprocess or path errors in notebook context

### US-009: E2E test — `run_training_job()` from Databricks notebook
**Description:** As a user in a Databricks notebook, I want to trigger a deployed training job using the Python API.

**Acceptance Criteria:**
- [ ] Prerequisite: a job is already deployed from US-007
- [ ] Notebook imports `from az_databricks_mlops import run_training_job`
- [ ] Calls `run_training_job("<target>-<project>-model-training-job")`
- [ ] Job is triggered successfully (prints run ID and monitor URL)
- [ ] No import errors or SDK authentication issues

### US-010: Automated test — `adm document` snapshot filtering
**Description:** As a developer, I want pytest coverage for the gitignore and low-signal filtering so regressions are caught.

**Acceptance Criteria:**
- [ ] Existing test `test_collect_repository_snapshot_respects_gitignore` passes (already added in 174cd57)
- [ ] Existing test `test_collect_repository_snapshot_ignores_low_signal_generated_artifacts` passes (already added in 84767ef)
- [ ] Add a test that verifies binary files (`.pkl`, `.parquet`, `.pt`) are excluded from snapshots
- [ ] Add a test that verifies the `--max-file-chars` truncation works correctly
- [ ] All tests pass: `pytest tests/ -q`

## Functional Requirements

- FR-1: README must document all 7 `adm` commands with consistent format: description, usage example, options list
- FR-2: README must include a "Notebook Usage" section showing programmatic API import patterns
- FR-3: README must mention smart filtering behavior (gitignore respect, low-signal artifact exclusion) under `adm document`
- FR-4: `adm document` must produce valid output when given a local path, remote Git URL, or no source (current directory)
- FR-5: `adm document` must work when called programmatically from a Databricks notebook via `review_repository()`
- FR-6: `run_training_job()` must work from Databricks notebooks with ambient workspace credentials
- FR-7: The full pipeline (`init` -> `run` -> job completes -> `trigger` re-runs) must succeed against the dev target
- FR-8: Automated pytest tests must cover snapshot filtering (gitignore, binary exclusion, low-signal artifacts, truncation)

## Non-Goals

- No changes to the `adm init` or `adm new` template generation logic
- ~~No new CLI flags or commands~~ — `--no-validate` was added to `adm init` per Decision #2
- No CI/CD pipeline setup (tests are run manually with real Databricks endpoints)
- No changes to the LLM prompt or review quality — only validating that the plumbing works
- No support for `adm init` / `adm run` inside notebooks (these require `databricks bundle` CLI which isn't in notebook runtimes)
- No automated notebook test harness — notebook tests are manual verification

## Technical Considerations

- **Databricks credentials**: E2E tests (US-005 through US-009) require a configured Databricks profile with access to Model Serving endpoints. The `.env` file should have `DATABRICKS_CONFIG_PROFILE` set.
- **Git availability in notebooks**: Databricks Runtime includes `git` in PATH, so `subprocess.run(["git", ...])` works. The `review_repository()` function should work as-is.
- **Notebook working directory**: Notebooks start in `/databricks/driver/`. The `review_repository()` call must use absolute paths for `source` and `output_path` to avoid confusion.
- **`adm trigger` vs `adm run` in notebooks**: `trigger` works (SDK-only). `run` does NOT work (needs `databricks bundle` CLI). README must make this distinction clear.
- **Remote Git URL for testing**: The 10x sibling repos use SSH remotes (e.g. `git@github-work:...`). These work if the developer's SSH keys are configured. For notebook testing, use HTTPS or a pre-cloned local path.
- **Endpoint availability**: Use `--endpoint databricks-gpt-5-4-mini` for testing (cheapest/fastest). The auto-selection prefers more capable models.

## Success Metrics

- README accurately describes 100% of CLI surface (7/7 commands, notebook API, filtering behavior)
- `adm document` succeeds against local path AND remote Git URL
- Full MLOps pipeline completes successfully in Databricks (job status: SUCCEEDED)
- `review_repository()` and `run_training_job()` both work from a Databricks notebook
- `pytest tests/ -q` passes with no failures

## Decisions (closed)

1. **Remote Git URL**: Use either a 10x sibling project via SSH or a small public ML repo — whichever is reachable at test time.
2. **`--no-validate` flag**: Yes — add `--no-validate` to `adm init` so it can run in notebook environments where `databricks bundle` CLI is unavailable. Update Non-Goals accordingly.
3. **Notebook testing**: Build an automated test notebook deployed as a Databricks job for repeatable validation (not one-time manual).
