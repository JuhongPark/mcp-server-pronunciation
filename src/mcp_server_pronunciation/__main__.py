"""Allow running as python -m mcp_server_pronunciation."""

import sys

from .cli import main

raise SystemExit(main(sys.argv[1:]))
