# AI Use and Agent Guidelines

This project may be worked on by several teammates using different AI assistants or coding agents. These rules are intended to reduce accidental damage, maintain academic integrity, and keep contributions reviewable.

## Core principle

AI is a development assistant. It is not the owner of the repository and should not make irreversible or high-impact decisions without human review.

## Required workflow for AI-assisted changes

1. Work on a dedicated branch.
2. Read `AGENTS.md` before editing.
3. Inspect existing implementation before proposing replacements.
4. Keep changes focused on one task or requirement.
5. Open a pull request.
6. Explain what changed and what was checked.
7. Have a team member review the work before merging.

## Changes that require explicit human confirmation

Agents should stop and ask before:

- deleting or renaming major folders
- removing substantial teammate work
- replacing an established analysis approach
- changing the project scope
- changing source data
- force-pushing or rewriting Git history
- merging into `main`
- adding paid services or infrastructure
- publishing private data
- introducing credentials or secrets

## Avoiding agent collisions

When multiple teammates use agents:

- use separate branches per task
- avoid having two agents edit the same file for unrelated work at the same time
- check recent commits and open pull requests before starting broad changes
- keep pull requests small and merge frequently enough to reduce divergence
- rebase or update your branch before final review when needed
- resolve conflicts manually and verify the resulting code

## Academic integrity

The team remains responsible for the submitted work.

AI-assisted work should be reviewed and understood by team members. Where IT5006 or NUS rules require disclosure of AI usage, the team should disclose it appropriately.

Do not use AI to fabricate or misrepresent:

- dataset contents
- statistical findings
- model results or metrics
- experiments that were not run
- references or citations
- screenshots or execution evidence
- team member contributions
- testing or validation that did not happen

## Suggested AI usage record

For material AI-assisted contributions, the pull request description can record:

- tool used
- purpose of assistance
- files materially affected
- human review performed

This keeps the history understandable without requiring every small prompt to be recorded.

## Human ownership

Before merging assessed work, at least one team member should be able to explain:

- what the code or analysis does
- why the method was chosen
- what assumptions it makes
- how the output was validated
- what limitations remain
