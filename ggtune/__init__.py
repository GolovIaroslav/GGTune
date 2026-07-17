"""GGTune — automatic llama.cpp parameter optimizer."""
import sys

# Rich prints ✓/✗/⚠/→ throughout the UI. On Windows, Python's stdout/stderr
# encoding follows the system's non-Unicode ANSI codepage (e.g. cp1251 on a
# Russian locale) rather than the terminal's actual codepage, so those
# characters crash with UnicodeEncodeError even in a UTF-8 console. Force
# UTF-8 unconditionally before anything else in this package prints.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

__version__ = "0.1.0"
