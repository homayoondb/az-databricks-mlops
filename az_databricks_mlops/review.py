"""Repository review workflow powered by Databricks Model Serving."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

DEFAULT_MAX_FILE_CHARS = 120_000
DEFAULT_MAX_TOTAL_CHARS = 2_400_000
DEFAULT_MAX_OUTPUT_TOKENS = 12_000
INTERNAL_DIR_NAME = ".adm_internal"
RESEARCH_FILENAME = "databricks-mlops-review-research.md"

MODEL_ENDPOINT_PREFERENCES: tuple[str, ...] = (
    # 1M context
    "databricks-claude-opus-4-6",
    # 400K context, 128K output
    "databricks-gpt-5-4",
    "databricks-gpt-5-2",
    "databricks-gpt-5-2-codex",
    # 200K context
    "databricks-claude-opus-4-5",
    "databricks-claude-sonnet-4-6",
    # 128K context — cost-optimized fallbacks
    "databricks-gpt-5-4-mini",
    "databricks-gpt-5-mini",
)

PRUNE_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".terraform",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "__pycache__",
    "htmlcov",
    "node_modules",
    "dist",
    "build",
    INTERNAL_DIR_NAME,
}

LOW_SIGNAL_DIRECTORY_NAMES = {
    ".databricks",
    "logs",
    "log",
    "mlruns",
    "wandb",
}

SKIP_DIRECTORY_NAMES = PRUNE_DIRECTORY_NAMES | LOW_SIGNAL_DIRECTORY_NAMES

LOW_SIGNAL_FILE_NAMES = {
    ".coverage",
    "terraform.tfstate",
    "terraform.tfstate.backup",
}

SKIP_FILE_NAMES = {".DS_Store"} | LOW_SIGNAL_FILE_NAMES

SKIP_TEXT_FILE_SUFFIXES = {
    ".log",
}

BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bmp",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mp3",
    ".mp4",
    ".mov",
    ".npy",
    ".npz",
    ".o",
    ".otf",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pyc",
    ".pyd",
    ".pyo",
    ".so",
    ".tar",
    ".tif",
    ".tiff",
    ".ttf",
    ".wav",
    ".webp",
    ".woff",
    ".woff2",
    ".zip",
}

BEST_PRACTICE_CONTEXT = """Apply the following up-to-date Databricks and MLflow guidance while reviewing the repository:
- Databricks recommends separate development, staging, and production environments, Git-based promotion, MLflow tracking, Unity Catalog model lifecycle management, validation gates, monitoring, and retraining workflows.
- For classic ML repositories, prefer MLflow's classic evaluation system: mlflow.models.evaluate(...) plus mlflow.validate_evaluation_results(...) for thresholds.
- For agentic / LLM / RAG repositories, prefer MLflow 3 tracing, mlflow.genai.evaluate(...), evaluation datasets, scorers / LLM judges, production monitoring, and human feedback loops.
- Do not recommend legacy MLflow 2 Agent Evaluation for new builds unless explicitly needed for backwards compatibility.
- For large repo review workloads on Databricks Model Serving, prioritize powerful long-context endpoints such as Claude Opus 4.6, Gemini 3.1 Pro, Gemini 2.5 Pro, GPT-5.4, and GPT-5.3 Codex.
- Repositories should be evaluated on both general software quality and Databricks-specific MLOps / LLMOps readiness.

Reference sources reviewed on 2026-04-11:
1. Databricks-hosted foundation models available in Foundation Model APIs (last updated Apr 8, 2026): https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html
2. MLOps workflows on Databricks (last updated Dec 18, 2024): https://docs.databricks.com/en/machine-learning/mlops/mlops-workflow.html
3. Evaluate and monitor AI agents in MLflow 3 on Databricks (last updated Mar 3, 2026): https://docs.databricks.com/en/mlflow3/genai/eval-monitor/index.html
4. MLflow LLM and Agent Evaluation docs: https://mlflow.org/docs/latest/genai/eval-monitor/
5. MLflow classic model evaluation docs: https://mlflow.org/docs/latest/ml/evaluation/
"""

REVIEW_SYSTEM_PROMPT = """You are an expert Databricks MLOps, LLMOps, and software architecture reviewer.

Your task is to review a repository using ONLY the provided repository snapshot and produce an easy-to-navigate Markdown report for engineers and technical leaders.

Required behavior:
- Classify the repository as one of: software-only, data pipeline, classic ML, LLM app, RAG app, agentic app, or hybrid.
- Distinguish clearly between observed evidence and missing evidence.
- Apply both general engineering best practices and Databricks / MLflow best practices.
- If the repo appears agentic or LLM-oriented, use MLflow 3 style evaluation and monitoring expectations.
- If the repo appears classic ML-oriented, use classic MLflow evaluation and validation expectations.
- Rank recommendations by priority using impact, risk, implementation effort, and confidence.
- Keep the report actionable, concise, and easy to scan.
- Avoid overly detailed prose; give high-level step-by-step next actions.

Output format requirements:
1. A title heading.
2. Executive summary.
3. Repository classification.
4. A "Start here" section with the top 3-5 actions in priority order.
5. A weighted scorecard table with dimensions, score (0-5), rationale, and evidence.
6. Prioritized findings grouped into Now, Next, Later.
7. A short step-by-step improvement plan.
8. Missing information / assumptions.

For every finding include:
- Why it matters.
- Evidence with file paths.
- A concrete next step.
- Estimated effort (Low / Medium / High).
- Confidence (Low / Medium / High).
"""

INTERNAL_RESEARCH_MARKDOWN = f"""# ADM Databricks MLOps review research

Generated on 2026-04-11 for internal prompt design.

## What this is for

This note captures the references and review policy used by `adm document` so the repository-review prompt stays aligned with current Databricks, MLflow, and agent-evaluation guidance.

## Key decisions

- Prefer long-context Databricks-hosted serving endpoints for deep repo review.
- Exclude low-signal runtime artifacts such as MLflow run folders, generated Databricks bundle state, Terraform cache/state, and log directories so prompt budget stays focused on source code and configuration.
- Use a single Markdown report that is easy to scan and grouped into `Start here`, `Now`, `Next`, and `Later` priorities.
- Separate expectations for classic ML repos versus LLM / RAG / agentic repos.
- Treat missing evidence explicitly instead of assuming a capability exists or does not exist.

## Endpoint preference order

{os.linesep.join(f'- `{endpoint}`' for endpoint in MODEL_ENDPOINT_PREFERENCES)}

## Best-practice context used in the prompt

{BEST_PRACTICE_CONTEXT}

## Source list

- Databricks-hosted foundation models available in Foundation Model APIs — https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html
- MLOps workflows on Databricks — https://docs.databricks.com/en/machine-learning/mlops/mlops-workflow.html
- Evaluate and monitor AI agents — https://docs.databricks.com/en/mlflow3/genai/eval-monitor/index.html
- MLflow LLM and Agent Evaluation — https://mlflow.org/docs/latest/genai/eval-monitor/
- MLflow classic model evaluation — https://mlflow.org/docs/latest/ml/evaluation/
"""


@dataclass(frozen=True)
class CollectedFile:
    """A collected text file included in the repo review prompt."""

    path: str
    characters: int
    truncated: bool


@dataclass(frozen=True)
class OmittedFile:
    """A file omitted or truncated from the review prompt."""

    path: str
    reason: str


@dataclass(frozen=True)
class RepositorySnapshot:
    """Collected repository text that will be sent to the LLM."""

    repo_name: str
    source_label: str
    root_path: Path
    files: tuple[CollectedFile, ...]
    omitted_files: tuple[OmittedFile, ...]
    total_characters: int
    prompt_payload: str


@dataclass(frozen=True)
class ReviewArtifact:
    """Details about a generated review document."""

    repo_name: str
    source_label: str
    endpoint_name: str
    prompt_path: Path
    research_path: Path
    output_path: Path
    snapshot: RepositorySnapshot


@dataclass(frozen=True)
class _ResolvedRepository:
    """Repository source materialized as a local directory."""

    repo_name: str
    source_label: str
    root_path: Path


@contextmanager
def _materialize_repository(source: str | None, working_directory: Path) -> Iterator[_ResolvedRepository]:
    """Resolve a local directory or clone a remote GitHub repository to a temp dir."""
    if source is None or source.strip() in {"", "."}:
        yield _ResolvedRepository(
            repo_name=working_directory.name,
            source_label=str(working_directory),
            root_path=working_directory,
        )
        return

    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = (working_directory / candidate).resolve()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Repository source must be a directory: {candidate}")
        yield _ResolvedRepository(
            repo_name=candidate.name,
            source_label=str(candidate),
            root_path=candidate,
        )
        return

    if not _looks_like_git_url(source):
        raise ValueError(f"Repository source does not exist and is not a supported Git URL: {source}")

    repo_name = _repo_name_from_source(source)
    with tempfile.TemporaryDirectory(prefix="adm-document-") as tmp_dir:
        clone_target = Path(tmp_dir) / repo_name
        result = subprocess.run(
            ["git", "clone", "--depth", "1", source, str(clone_target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip() or "unknown git clone failure"
            raise ValueError(f"Failed to clone {source}: {message}")
        yield _ResolvedRepository(
            repo_name=repo_name,
            source_label=source,
            root_path=clone_target,
        )


def _looks_like_git_url(value: str) -> bool:
    """Return whether the input looks like a cloneable Git URL."""
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return True
    return value.startswith("git@") and ":" in value


def _repo_name_from_source(source: str) -> str:
    """Infer a human-readable repository name from a local path or remote URL."""
    if source.startswith("git@"):
        tail = source.split(":", 1)[1]
    else:
        tail = urlparse(source).path or source
    name = Path(tail).name or "repository"
    if name.endswith(".git"):
        name = name[: -len(".git")]
    return name or "repository"


def collect_repository_snapshot(
    root_path: Path,
    *,
    source_label: str,
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> RepositorySnapshot:
    """Collect repository text into a single prompt payload."""
    collected_files: list[CollectedFile] = []
    omitted_files: list[OmittedFile] = []
    prompt_sections: list[str] = []
    total_characters = 0

    for file_path in _iter_repo_files(root_path):
        relative_path = file_path.relative_to(root_path).as_posix()
        skip_reason = _skip_reason_for_relative_path(relative_path)
        if skip_reason is not None:
            omitted_files.append(OmittedFile(path=relative_path, reason=skip_reason))
            continue

        text, reason = _read_text_file(file_path)
        if text is None:
            omitted_files.append(OmittedFile(path=relative_path, reason=reason))
            continue

        truncated = False
        if len(text) > max_file_chars:
            text = text[:max_file_chars]
            truncated = True
            omitted_files.append(OmittedFile(path=relative_path, reason=f"truncated to first {max_file_chars} characters"))

        remaining_budget = max_total_chars - total_characters
        if remaining_budget <= 0:
            omitted_files.append(OmittedFile(path=relative_path, reason="omitted after reaching total prompt budget"))
            continue
        if len(text) > remaining_budget:
            text = text[:remaining_budget]
            truncated = True
            omitted_files.append(OmittedFile(path=relative_path, reason="partially included because the overall prompt budget was reached"))

        prompt_sections.append(
            "\n".join(
                [
                    f"## FILE: {relative_path}",
                    "```",
                    text,
                    "```",
                ]
            )
        )
        total_characters += len(text)
        collected_files.append(
            CollectedFile(
                path=relative_path,
                characters=len(text),
                truncated=truncated,
            )
        )

    manifest_lines = [f"- {item.path} ({item.characters} chars{' , truncated' if item.truncated else ''})".replace(" ,", ",") for item in collected_files]
    omitted_lines = [f"- {item.path}: {item.reason}" for item in omitted_files] or ["- None"]
    payload = "\n\n".join(
        [
            f"Repository root: {root_path}",
            f"Source: {source_label}",
            f"Collected text files: {len(collected_files)}",
            f"Prompt characters: {total_characters}",
            "\nIncluded files manifest:\n" + ("\n".join(manifest_lines) if manifest_lines else "- None"),
            "\nOmitted or truncated files:\n" + "\n".join(omitted_lines),
            "\nRepository contents:\n" + ("\n\n".join(prompt_sections) if prompt_sections else "<no text files collected>"),
        ]
    )

    return RepositorySnapshot(
        repo_name=root_path.name,
        source_label=source_label,
        root_path=root_path,
        files=tuple(collected_files),
        omitted_files=tuple(omitted_files),
        total_characters=total_characters,
        prompt_payload=payload,
    )


def _skip_reason_for_relative_path(relative_path: str) -> str | None:
    """Return an omission reason when a path is a low-signal generated artifact."""
    path = PurePosixPath(relative_path)

    for part in path.parts[:-1]:
        if part in SKIP_DIRECTORY_NAMES or part.endswith(".egg-info"):
            return f"ignored irrelevant directory '{part}'"

    file_name = path.name
    if file_name in LOW_SIGNAL_FILE_NAMES or file_name.startswith(".coverage"):
        return f"ignored irrelevant generated file '{file_name}'"

    if path.suffix.lower() in SKIP_TEXT_FILE_SUFFIXES:
        return f"ignored irrelevant log or output file '{file_name}'"

    return None


def _iter_repo_files(root_path: Path) -> Iterator[Path]:
    """Yield candidate files in deterministic order."""
    yielded = False
    git_dir = root_path / ".git"
    if git_dir.exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for relative_path in sorted(filter(None, result.stdout.splitlines())):
                file_path = root_path / relative_path
                if file_path.is_file() and not file_path.is_symlink():
                    yielded = True
                    yield file_path
    if yielded:
        return

    for current_root, dirnames, filenames in os.walk(root_path):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in PRUNE_DIRECTORY_NAMES and not name.endswith(".egg-info")
        )
        for filename in sorted(filenames):
            if filename == ".DS_Store":
                continue
            file_path = Path(current_root) / filename
            if file_path.is_symlink():
                continue
            yield file_path


def _read_text_file(file_path: Path) -> tuple[str | None, str]:
    """Return decoded text or an omission reason for a file."""
    if file_path.suffix.lower() in BINARY_SUFFIXES:
        return None, "binary or unsupported extension"

    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        return None, f"failed to read file: {exc}"

    if b"\x00" in raw:
        return None, "binary content"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - defensive only
            return None, f"failed to decode text: {exc}"

    return text, ""


def select_review_endpoint(workspace_client: WorkspaceClient, preferred_endpoint: str | None = None) -> str:
    """Choose the best available Databricks Model Serving endpoint for repo review."""
    endpoints = list(workspace_client.serving_endpoints.list())
    ready_endpoints = {
        endpoint.name
        for endpoint in endpoints
        if _is_ready(endpoint) and getattr(endpoint, "name", None)
    }
    if not ready_endpoints:
        raise ValueError("No READY Databricks Model Serving endpoints were found.")

    if preferred_endpoint is not None:
        if preferred_endpoint not in ready_endpoints:
            available = ", ".join(sorted(ready_endpoints))
            raise ValueError(
                f"Requested endpoint '{preferred_endpoint}' is not READY. Available READY endpoints: {available}"
            )
        return preferred_endpoint

    for endpoint_name in MODEL_ENDPOINT_PREFERENCES:
        if endpoint_name in ready_endpoints:
            return endpoint_name

    return sorted(ready_endpoints)[0]


def _is_ready(endpoint: object) -> bool:
    """Return whether a serving endpoint is ready to accept queries."""
    state = getattr(endpoint, "state", None)
    ready_value = getattr(state, "ready", None)
    value = getattr(ready_value, "value", ready_value)
    return value == "READY"


def build_review_prompt(snapshot: RepositorySnapshot) -> str:
    """Build the user prompt for repository review."""
    return f"""Review the following repository snapshot and generate a prioritized improvement document.

{BEST_PRACTICE_CONTEXT}

Repository name: {snapshot.repo_name}
Repository source: {snapshot.source_label}
Collected files: {len(snapshot.files)}
Collected characters: {snapshot.total_characters}

Review requirements:
- Identify whether this is primarily a classic ML, data engineering, LLM, RAG, agentic, hybrid, or general software repository.
- Use a weighted scorecard that covers architecture clarity, developer experience, reproducibility, testing / CI, security / governance, maintainability, observability, and the most relevant ML / LLM / agentic dimensions.
- Prioritize findings so a time-constrained user can start immediately.
- Focus strongly on Databricks MLOps and MLflow best practices where relevant.
- Recommend concrete next steps in rank order.
- Keep the report complete but navigable.

Repository snapshot begins below.

{snapshot.prompt_payload}
"""


def query_review_model(
    workspace_client: WorkspaceClient,
    *,
    endpoint_name: str,
    user_prompt: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """Send the repository review prompt to Databricks Model Serving."""
    response = workspace_client.serving_endpoints.query(
        name=endpoint_name,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=REVIEW_SYSTEM_PROMPT),
            ChatMessage(role=ChatMessageRole.USER, content=user_prompt),
        ],
        max_tokens=max_output_tokens,
        temperature=0,
    )

    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content:
            return content.strip()
        text = getattr(choices[0], "text", None)
        if text:
            return text.strip()

    predictions = getattr(response, "predictions", None)
    if predictions:
        first_prediction = predictions[0]
        if isinstance(first_prediction, str):
            return first_prediction.strip()

    raise ValueError("Model serving response did not contain any text output.")


def ensure_internal_reference_files(working_directory: Path, prompt_text: str) -> tuple[Path, Path]:
    """Write the internal research note and the exact prompt used for the latest run."""
    internal_dir = working_directory / INTERNAL_DIR_NAME
    internal_dir.mkdir(parents=True, exist_ok=True)

    research_path = internal_dir / RESEARCH_FILENAME
    research_path.write_text(INTERNAL_RESEARCH_MARKDOWN)

    prompt_path = internal_dir / "latest-review-prompt.txt"
    prompt_path.write_text(prompt_text)

    return prompt_path, research_path


def review_repository(
    *,
    source: str | None,
    output_path: Path | None,
    working_directory: Path,
    preferred_endpoint: str | None = None,
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    workspace_client: WorkspaceClient | None = None,
) -> ReviewArtifact:
    """Generate a repository review document from a local or remote repository."""
    with _materialize_repository(source, working_directory) as resolved:
        snapshot = collect_repository_snapshot(
            resolved.root_path,
            source_label=resolved.source_label,
            max_file_chars=max_file_chars,
            max_total_chars=max_total_chars,
        )
        prompt_text = build_review_prompt(snapshot)
        prompt_path, research_path = ensure_internal_reference_files(working_directory, prompt_text)
        client = workspace_client or WorkspaceClient()
        endpoint_name = select_review_endpoint(client, preferred_endpoint)
        report_body = query_review_model(
            client,
            endpoint_name=endpoint_name,
            user_prompt=prompt_text,
        )

        destination = output_path or (working_directory / f"{resolved.repo_name}-adm-review.md")
        if not destination.is_absolute():
            destination = (working_directory / destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        header = "\n".join(
            [
                "<!-- Generated by adm document -->",
                f"<!-- Source: {resolved.source_label} -->",
                f"<!-- Review endpoint: {endpoint_name} -->",
                f"<!-- Generated at: {generated_at} -->",
                "",
            ]
        )
        destination.write_text(header + report_body.strip() + "\n")

        return ReviewArtifact(
            repo_name=resolved.repo_name,
            source_label=resolved.source_label,
            endpoint_name=endpoint_name,
            prompt_path=prompt_path,
            research_path=research_path,
            output_path=destination,
            snapshot=snapshot,
        )
