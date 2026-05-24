import json
import logging
from datetime import datetime, timezone
import sys

logger = logging.getLogger("inferr.debug")

class DebugLogger:
    def __init__(self):
        self.enabled = True

    def _format_timestamp(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def log_stage(self, stage: str, data: dict, print_to_console: bool = True):
        if not self.enabled:
            return

        log_entry = {
            "timestamp": self._format_timestamp(),
            "stage": stage,
            "data": data
        }
        
        # We can write to a file or stdout
        if print_to_console:
            print(f"\n{'='*50}\n[STAGE: {stage}] ({log_entry['timestamp']})\n{'='*50}", file=sys.stderr)
            
            # Format nicely
            for key, val in data.items():
                if isinstance(val, str) and len(val) > 100:
                    print(f"\n--- {key.upper()} ---\n{val}\n-------------------", file=sys.stderr)
                else:
                    try:
                        print(f"{key}: {json.dumps(val, indent=2)}", file=sys.stderr)
                    except TypeError:
                        print(f"{key}: {val}", file=sys.stderr)
            print(f"{'='*50}\n", file=sys.stderr)

debug_logger = DebugLogger()
