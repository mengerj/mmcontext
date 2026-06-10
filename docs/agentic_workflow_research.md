# Agentic Workflow Plan: mmcontext Repository Automation

## Decision: Claude-native approach

We're going all-in on the Claude ecosystem: Claude Code Action for GitHub automation, gh-aw for scheduled audits, Cowork for interactive work. No multi-provider abstraction for now.

## Current State

- Repo: `github.com/mengerj/mmcontext` — multimodal contrastive learning (text + omics) via sentence-transformers
- Existing workflows: `test.yaml`, `build.yaml`, `release.yaml`
- Branch strategy: `dev-claude` exists as the integration branch for agent work
- Auth: Need to run `claude setup-token` and store OAuth token as repo secret
- No `CLAUDE.md`, no benchmarks, no agent workflows yet

---

## Phase 1: Foundation — CLAUDE.md

**What**: Create `CLAUDE.md` in repo root with project context, conventions, and instructions for agent behavior.

**Who does what**:
- **Claude (Cowork)**: Drafts the file based on repo analysis
- **Jo**: Reviews, adjusts tone/priorities, commits to `dev-claude`

**Contents**: Package structure, key modules, test/lint commands, branch conventions (always branch from `dev-claude`), PR conventions (link to issues), coding standards (ruff, type hints, docstrings).

---

## Phase 2: Authentication Setup

**What**: Generate OAuth token for subscription-based GitHub Actions.

**Who does what**:
- **Jo** (manual, ~5 min):
  1. Run `claude setup-token` in terminal
  2. Go to GitHub repo → Settings → Secrets → Actions
  3. Add secret `CLAUDE_CODE_OAUTH_TOKEN` with the token value

**Note**: Token expires if you log out of Claude Code. Regenerate with `claude setup-token` and update the secret. Starting June 15, 2026, programmatic usage draws from a separate credit pool capped at your subscription fee.

---

## Phase 3: PR Review Workflow

**What**: Automated code review on every PR targeting `dev-claude` or `main`.

**Who does what**:
- **Claude (Cowork)**: Drafts `.github/workflows/claude-review.yaml`
- **Jo**: Reviews, commits, tests with a real PR

**Workflow triggers**: PR opened/updated against `dev-claude` or `main`, plus `@claude` mentions in PR comments.

**Key config**:
- OAuth token auth (no API key)
- `actions: read` permission so Claude can see CI results
- Custom instructions pointing to `CLAUDE.md`
- Allowed tools: `Bash(pytest)`, `Bash(ruff check)`

---

## Phase 4: Issue-to-PR Workflow (plan-first protocol)

**What**: When you mention `@claude` on an issue, the agent:
1. Posts clarifying questions if the issue is ambiguous
2. Posts an implementation plan as a checkbox list in a comment
3. Waits for your approval (`@claude approved` or `@claude go ahead`)
4. Creates a branch from `dev-claude`, implements the plan, ticks checkboxes as it goes
5. Opens a PR linked to the issue

**Who does what**:
- **Claude (Cowork)**: Drafts `.github/workflows/claude-implement.yaml` and the plan-first protocol in `CLAUDE.md`
- **Jo**: Reviews, commits, tests with a real issue

**Branch naming**: `claude/<issue-number>-<short-slug>` (e.g., `claude/47-add-benchmark-scripts`)

**Issue linking**: Claude Code Action can include `Fixes #47` or `Closes #47` in PR descriptions, which GitHub auto-links. The CLAUDE.md instructions will enforce this convention.

**Plan-first enforcement**: Encoded in `CLAUDE.md` as the "Issue Implementation Protocol" — the agent reads this on every run and follows the plan-first flow.

**First test**: Create an issue for the benchmark scripts (Phase 6) and let `@claude` implement it. This gives us a real test of the workflow AND gets the benchmarks built.

---

## Phase 5: Benchmark Scripts

**What**: pytest-benchmark scripts measuring the three priority metrics:
1. **Peak GPU memory** — `torch.cuda.max_memory_allocated()` during forward/training
2. **Embedding quality** — kNN accuracy, retrieval metrics from `mmcontext.eval`
3. **Training throughput** — samples/sec during a training loop

**Who does what — two options**:
- **Option A (GitHub workflow test)**: Create a GitHub issue with benchmark requirements, `@claude` it, and let the agent build the scripts. Good test of Phase 4.
- **Option B (Cowork)**: Build here interactively if the GitHub workflow isn't set up yet or the task proves too complex for headless execution.

**Recommendation**: Try Option A first. If the agent struggles (benchmarks need domain knowledge about mmcontext's model/data pipeline), fall back to Option B.

**GPU constraint**: GitHub Actions runners lack GPUs. The benchmark workflow will run CPU-only smoke tests for regression detection. Real GPU benchmarks run locally or via SLURM (you already have `.slurm` scripts). Results can be pushed to the same tracking system.

---

## Phase 6: Benchmark GitHub Action

**What**: `.github/workflows/benchmark.yaml` — runs benchmarks on push to `main`/`dev-claude`, weekly schedule, and tracks results via [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark).

**Who does what**:
- **Claude (Cowork or @claude)**: Drafts the workflow
- **Jo**: Reviews, enables GitHub Pages for the benchmark chart

**Features**: Publishes charts to GitHub Pages, comments on PRs when performance regresses >150%, stores historical data in `gh-pages` branch.

---

## Phase 7: gh-aw Scheduled Audits

**What**: Use [GitHub Agentic Workflows](https://github.github.com/gh-aw/) for recurring, automated codebase analysis. Workflows are markdown files that compile to GitHub Actions YAML with built-in security guardrails (read-only tokens, network firewall, threat detection).

**Who does what**:
- **Jo** (manual, ~10 min):
  1. `gh extension install github/gh-aw`
  2. `gh aw init` in the repo
  3. Set up engine secrets (Claude via OAuth)
- **Claude (Cowork)**: Drafts workflow markdown files
- **Jo**: Reviews, merges, monitors first runs

**Planned workflows**:

### 7a. Weekly Code Quality Audit
```markdown
<!-- .github/workflows/code-quality-audit.md -->
---
name: Code Quality Audit
on:
  schedule: weekly on mondays
engine: claude
safe-outputs:
  create-discussion:
    title-prefix: "[audit] "
    category: "audits"
    max: 1
tools:
  cache-memory: true
  github:
    toolsets: [default]
---

Analyze the mmcontext codebase for:
- Test coverage gaps in src/mmcontext/
- Type annotation completeness
- Memory efficiency in data loading (io/, embed/)
- API compatibility with sentence-transformers v5.4
- Dead code in _legacy/ that should be removed

Compare against previous audit findings in cache memory.
Create a discussion with prioritized recommendations.
```

### 7b. Test Coverage Monitor
Runs on push, checks what changed, flags untested code paths.

### 7c. ResearchPlanAssignOps cycle
The audit discussion feeds into `/plan` → sub-issues → `@claude` implements. Full loop.

**gh-aw engine options**: Can use Claude, Copilot, or Codex as engine — useful for future comparison without building custom abstractions.

---

## Phase 8: Cowork Scheduled Tasks — Comparison

**What**: Set up a Cowork scheduled task that does similar work to the gh-aw audit, then compare the two approaches.

**Differences**:
| Aspect | gh-aw | Cowork Scheduled Task |
|--------|-------|-----------------------|
| Runs where | GitHub Actions (cloud) | Local (Cowork must be open) |
| Security | 5-layer guardrails, read-only token | Full local access |
| Output | GitHub Discussions/Issues | Chat notification in Cowork |
| Data access | Repo only (sandboxed) | Full local filesystem |
| Scheduling | Cron via Actions | Cron via Cowork app |
| Cost | GitHub Actions minutes + LLM tokens | Subscription credit pool |

**Verdict**: gh-aw for anything that should run reliably regardless of whether your laptop is open. Cowork tasks for things that need local context (your `.venv`, local data files, GPU) or personal briefings.

---

## Implementation Order

| Step | What | Who | Blocked by |
|------|------|-----|------------|
| **1** | Create `CLAUDE.md` | Claude (Cowork) → Jo reviews | — |
| **2** | Run `claude setup-token`, add repo secret | Jo (manual) | — |
| **3** | Draft + commit `claude-review.yaml` | Claude (Cowork) → Jo reviews | 1, 2 |
| **4** | Draft + commit `claude-implement.yaml` | Claude (Cowork) → Jo reviews | 1, 2 |
| **5** | Test: create benchmark issue, `@claude` it | Jo triggers, Claude implements | 3, 4 |
| **6** | Draft + commit `benchmark.yaml` | Claude (via @claude or Cowork) | 5 |
| **7** | Install gh-aw, set up audit workflows | Jo installs → Claude drafts | 2 |
| **8** | Set up Cowork scheduled task, compare | Claude (Cowork) | 7 |

Steps 1–2 can happen in parallel. Steps 3–4 can happen in parallel after 1–2.

---

## Settings Jo Needs to Configure

- [ ] `claude setup-token` → store as `CLAUDE_CODE_OAUTH_TOKEN` repo secret
- [ ] Enable GitHub Pages on the repo (for benchmark charts) — Settings → Pages → Source: `gh-pages` branch
- [ ] `gh extension install github/gh-aw` + `gh aw init`
- [ ] Set up gh-aw engine secret for Claude (see [engine docs](https://github.github.com/gh-aw/reference/engines/))
- [ ] Consider adding `pytest-benchmark` to `[project.optional-dependencies.test]` in `pyproject.toml`
- [ ] Decide on GitHub Discussions category "audits" (create it in repo Settings → Discussions)

---

## Key Links

- [Claude Code GitHub Actions docs](https://code.claude.com/docs/en/github-actions)
- [claude-code-action repo](https://github.com/anthropics/claude-code-action)
- [Claude Code Action with OAuth (Marketplace)](https://github.com/marketplace/actions/claude-code-action-with-oauth)
- [Claude Code Routines docs](https://code.claude.com/docs/en/routines)
- [claude-code-security-review](https://github.com/anthropics/claude-code-security-review)
- [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark)
- [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark)
- [GitHub Agentic Workflows](https://github.github.com/gh-aw/)
- [gh-aw ResearchPlanAssignOps](https://github.github.com/gh-aw/patterns/research-plan-assign-ops/)
- [June 2026 billing split](https://gist.github.com/yurukusa/7d854616809e673ca8d23353ed8267a6)
