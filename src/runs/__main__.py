"""Allow ``python -m runs`` invocation."""
from .cli import main
import sys

sys.exit(main())
