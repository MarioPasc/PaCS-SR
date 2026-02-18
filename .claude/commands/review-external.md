---
description: "Review an external repo's code against its paper and produce an insights document"
---

Invoke the external-code-reviewer subagent for the paper and repo specified.

**Format:** `/review-external <paper_folder_name> <external_repo_name>`
**Example:** `/review-external meanflow_2025 MeanFlow`

The subagent will:
1. Read the paper PDF from `docs/papers/$ARGUMENTS` (first argument is paper folder)
2. Read the code in `src/external/` (second argument is repo name)
3. Produce `docs/papers/{paper_folder}/insights.md` with 7 structured sections

Use the Task tool to launch the `external-code-reviewer` subagent with the arguments: $ARGUMENTS
