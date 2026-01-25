---
description: "Set up LatchLine config in ~/.latchline/settings.conf"
allowed-tools: ["Bash"]
---

# LatchLine Setup

Runs the LatchLine config bootstrap script. This creates `~/.latchline/settings.conf`
from the template in the plugin, if it does not already exist.

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/setup-config.py
```
