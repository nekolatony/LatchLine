---
description: "Set up LatchLine config (flags: --backend codex|claude|both, --block 0|1|2, --log-dir PATH)"
argument-hint: "[--backend codex|claude|both] [--block 0|1|2] [--log-dir PATH]"
allowed-tools: ["Bash"]
---

# LatchLine Setup

Runs the LatchLine config bootstrap script. This overwrites `~/.latchline/settings.conf`
from the template in the plugin, applying any overrides you pass.

```!
uv run --project ${CLAUDE_PLUGIN_ROOT} python ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/setup-config.py $ARGUMENTS
```
