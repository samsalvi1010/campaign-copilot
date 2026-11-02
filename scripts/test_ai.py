"""Quick check that the Gemini API works, and a clear explanation if it doesn't.

Run with:  python scripts/test_ai.py

It checks, step by step:
  1. the key can be read from .streamlit/secrets.toml (the key itself is NEVER printed)
  2. this computer can find Google's API server (DNS) and connect to it (network)
  3. a one-line prompt to Gemini works, and if not, it prints the full error
"""
import os
import socket
import sys
import traceback
from pathlib import Path

# Make "from src..." work when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib  # noqa: E402  (built into Python 3.11+)
    from google import genai  # noqa: E402
    from google.genai import types  # noqa: E402
except ImportError:
    # Usually means the script was run with a different Python (e.g. Anaconda's "base")
    sys.exit(f"google-genai isn't installed in this Python ({sys.executable}).\n"
             "Activate the project environment first:\n"
             "    source .venv/bin/activate\n"
             "    python scripts/test_ai.py")

from src.ai import MODEL_NAME  # noqa: E402

API_HOST = "generativelanguage.googleapis.com"
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"


def hide_key(text, key):
    """Replace the key with *** in any text, in case an error message contains it."""
    return text.replace(key, "***") if key else text


# --- Step 1: read the key --------------------------------------------------------
print("1. API key")
if not SECRETS_FILE.exists():
    sys.exit(f"   FAIL: {SECRETS_FILE} not found. Copy secrets.toml.example and add your key.")

try:
    key = tomllib.loads(SECRETS_FILE.read_text())["GEMINI_API_KEY"]
except (tomllib.TOMLDecodeError, KeyError) as error:
    sys.exit(f"   FAIL: could not read GEMINI_API_KEY from secrets.toml ({type(error).__name__}).")

if not key or key == "paste-your-key-here":
    sys.exit("   FAIL: GEMINI_API_KEY is empty or still the placeholder.")
# Describe the key's format without revealing it
problems = []
if key != key.strip():
    problems.append("has spaces or line breaks at the start/end")
if '"' in key or "'" in key:
    problems.append("contains quote characters")
print(f"   OK: key found ({len(key)} characters)" + (f", but it {', '.join(problems)}" if problems else ""))

# --- Step 2: network checks ------------------------------------------------------
print(f"\n2. Network: can this computer reach {API_HOST}?")
proxies = {name: value for name, value in os.environ.items() if "proxy" in name.lower()}
print(f"   Proxy settings: {proxies if proxies else 'none'}")

try:
    address = socket.getaddrinfo(API_HOST, 443)[0][4][0]
    print(f"   OK: DNS lookup works ({API_HOST} -> {address})")
except socket.gaierror as error:
    print(f"   FAIL: DNS lookup failed: {error}")
    address = None

if address:
    try:
        socket.create_connection((API_HOST, 443), timeout=10).close()
        print("   OK: connected to port 443 (HTTPS)")
    except OSError as error:
        print(f"   FAIL: could not connect to port 443: {error!r}")

# --- Step 3: a real one-line request ----------------------------------------------
print(f"\n3. Gemini request (model: {MODEL_NAME})")
try:
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30_000))
    response = client.models.generate_content(model=MODEL_NAME, contents="Reply with just: OK")
    print(f"   OK: Gemini replied: {response.text.strip()!r}")
except Exception as error:
    print(f"   FAIL: {type(error).__name__}\n")
    print("   Full error (key hidden):")
    print(hide_key("".join(traceback.format_exception(error)), key))
    # Follow the chain of causes: the root cause is usually at the bottom
    cause = error.__cause__ or error.__context__
    while cause is not None:
        print(f"   Caused by: {type(cause).__name__}: {hide_key(str(cause), key)}")
        cause = cause.__cause__ or cause.__context__
    sys.exit(1)
