---
description: "Implement a specific project phase end-to-end"
---

Implement phase $ARGUMENTS of the NeuroMF project.

**Format:** `/implement-phase <phase_number>`
**Example:** `/implement-phase 0`

Before invoking the phase-implementer:
1. Check that all previous phases have passing verification reports by reading `experiments/phase_{N-1}/verification_report.md` for each prior phase.
2. If any prior phase gate is BLOCKED, stop and report which phase is blocking.

If all gates are OPEN (or this is Phase 0), use the Task tool to launch the `phase-implementer` subagent with the phase number: $ARGUMENTS
