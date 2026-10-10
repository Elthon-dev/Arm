#!/usr/bin/env python3
"""
Mr. Braincells Type 4 — optimized one-file two-brain automation CLI.

Modes:
  python3 braincells_type2_v4.py                 # MAIN client
  python3 braincells_type2_v4.py --worker        # TOOL WIELDER HTTP server
  python3 braincells_type2_v4.py --both          # remote two-brain client

Fast overrides:
  python3 braincells_type2_v4.py --both --key 'KEY_HERE'
  python3 braincells_type2_v4.py --both --ip

--key overrides TOOL_WIELDER_KEY for this run and is never saved or printed.
--ip interactively selects Main/Worker and saves only the selected IPv4 address.

Main Brain:
  Gemma 4 12B / Ollama

Tool Brain:
  Qwen3 1.7B / Ollama

The Tool Wielder owns execution, files, web, and memory.
Gemma owns reasoning, code generation, diagnosis, and repair.

No third-party Python packages are required.

v4 optimization notes (see git-less diff against v3):
  SPEED   parallel consecutive read-only tool calls, parallel search backends,
          streamed final answers, DNS lookup cache, no duplicate SSRF validation,
          ripgrep --json fast path, hoisted realpath calls, prewarm on by default.
  QUALITY context-budget history trimming, uncapped tool-round output (no more
          truncated write_file JSON), IDF-weighted memory retrieval, hardened
          system prompts, hard round ceiling with a graceful final synthesis.
"""
from __future__ import annotations

import argparse
import html
import getpass
import hmac
import ipaddress
import json
import math
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import hashlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPSConnection
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urljoin
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError


# ============================================================
# CONFIG
# ============================================================

CONFIG_DIR = os.path.expanduser("~/.config/mr-braincells")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_MAIN_IP = "100.106.190.127"
DEFAULT_TOOL_IP = "100.117.192.80"
MAIN_PORT = 11434


def _load_endpoint_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _normalize_main_host(value: str) -> str:
    value = str(value).strip()
    if not value:
        return f"{DEFAULT_MAIN_IP}:{MAIN_PORT}"
    if value.count(":") == 0:
        return f"{value}:{MAIN_PORT}"
    return value


def _env_int(name: str, default: int, *aliases: str) -> int:
    """Read an int-typed env var without ever exploding the whole CLI at import time.

    A stray `MAIN_CTX=8k` used to raise ValueError and kill the process before the REPL
    even started. Now a bad value silently falls back to the default (and to any alias).
    """
    for key in (name,) + aliases:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            continue
    return default


def _env_float(name: str, default: float, *aliases: str) -> float:
    for key in (name,) + aliases:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _env_str(name: str, default: str, *aliases: str) -> str:
    for key in (name,) + aliases:
        raw = os.getenv(key)
        if raw:
            return raw
    return default


def _env_bool(name: str, default: bool, *aliases: str) -> bool:
    for key in (name,) + aliases:
        raw = os.getenv(key)
        if raw is None:
            continue
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return default


_ENDPOINT_CONFIG = _load_endpoint_config()
MAIN_HOST = _normalize_main_host(
    os.getenv("MAIN_HOST") or _ENDPOINT_CONFIG.get("main_host") or DEFAULT_MAIN_IP
)
MAIN_MODEL = _env_str("MAIN_MODEL", "gemma4:12b")
MAIN_KEEP_ALIVE = _env_str("MAIN_KEEP_ALIVE", "24h")
MAIN_CTX = _env_int("MAIN_CTX", 8192)
MAIN_MAX_OUTPUT = _env_int("MAIN_MAX_OUTPUT", 8192)
MAIN_TIMEOUT = _env_float("MAIN_TIMEOUT", 300.0)
# num_predict is a CAP, not a target: the model stops at EOS, so a generous cap
# costs nothing on short answers and prevents truncated tool-call JSON on long ones.
MAIN_TEMPERATURE = _env_float("MAIN_TEMPERATURE", 0.1)
SYNTH_TEMPERATURE = _env_float("SYNTH_TEMPERATURE", 0.2)

TOOL_HOST = os.getenv("TOOL_HOST") or str(_ENDPOINT_CONFIG.get("tool_host") or DEFAULT_TOOL_IP)
TOOL_PORT = _env_int("TOOL_PORT", 8765)
TOOL_WIELDER_KEY = os.getenv("TOOL_WIELDER_KEY", "")

QWEN_HOST = _env_str("QWEN_HOST", "127.0.0.1:11434", "OLLAMA_HOST")
QWEN_MODEL = _env_str("QWEN_MODEL", "qwen3:1.7b", "TOOL_MODEL")
QWEN_KEEP_ALIVE = _env_str("QWEN_KEEP_ALIVE", "24h", "OLLAMA_KEEP_ALIVE")
QWEN_CTX = _env_int("QWEN_CTX", 4096)
QWEN_TIMEOUT = _env_float("QWEN_TIMEOUT", 90.0)

WORKSPACE = os.path.abspath(os.getenv("WORKSPACE", os.getcwd()))
# realpath() walks the filesystem; computing it once instead of per-call (list_files used
# to call it once per directory entry) removes a measurable amount of syscall churn.
WORKSPACE_REAL = os.path.realpath(WORKSPACE)
MEMORY_FILE = os.path.abspath(
    os.getenv("MEMORY_FILE", os.path.join(WORKSPACE, "data", "tool_memory.jsonl"))
)
# Optional write-ahead drawer: every turn dropped by compaction is appended here so the
# memory distillery is lossy for the *prompt* but never lossy for the *user*.
MEMORY_DRAWER = os.path.abspath(
    os.getenv("MEMORY_DRAWER", os.path.join(os.path.expanduser("~"), ".mr-braincells", "memory_drawer.jsonl"))
)

COMMAND_TIMEOUT = _env_int("COMMAND_TIMEOUT", 180)
MAX_REQUEST_BODY = _env_int("MAX_REQUEST_BODY", 8 * 1024 * 1024)
MAX_FILE_READ = _env_int("MAX_FILE_READ", 512 * 1024)
MAX_WEB_BYTES = _env_int("MAX_WEB_BYTES", 1024 * 1024)
MAX_HISTORY = _env_int("MAX_HISTORY", 12)  # kept as whole conversation turns
MAX_TOOL_ROUNDS = _env_int("MAX_TOOL_ROUNDS", 0)  # 0 = adaptive/unbounded until the agent finishes
MAX_STALL_ROUNDS = _env_int("MAX_STALL_ROUNDS", 5)
# Absolute ceiling so a model that keeps making "successful" but useless calls can never hang forever.
MAX_ROUNDS_HARD = _env_int("MAX_ROUNDS_HARD", 40)
MAX_TOOL_RESULT = _env_int("MAX_TOOL_RESULT", 9000)
PREWARM = _env_bool("PREWARM", True)
CHAT_MAX_OUTPUT = _env_int("CHAT_MAX_OUTPUT", 4096)
CHAT_CTX = _env_int("CHAT_CTX", 4096)
TOOL_HISTORY_TURNS = _env_int("TOOL_HISTORY_TURNS", 3)
CHAT_HISTORY_TURNS = _env_int("CHAT_HISTORY_TURNS", 8)
SHOW_TIMINGS = os.getenv("BRAIN_TIMINGS", "0").lower() in {"1", "true", "yes"}
WEB_CACHE_TTL = _env_int("WEB_CACHE_TTL", 45)
WEB_TIMEOUT = _env_float("WEB_TIMEOUT", 12.0)
FETCH_TIMEOUT = _env_float("FETCH_TIMEOUT", 15.0)
SEARCH_BACKENDS = _env_str("SEARCH_BACKENDS", "ddg,bing")

STREAM = _env_bool("BRAIN_STREAM", True)
TOOL_PARALLEL = _env_bool("BRAIN_TOOL_PARALLEL", True)
TOOL_CONN_POOL = _env_int("TOOL_CONN_POOL", 6)
SEARCH_PARALLEL = _env_bool("BRAIN_SEARCH_PARALLEL", True)
DNS_CACHE_TTL = _env_float("DNS_CACHE_TTL", 30.0)
# Cap ONLY the TCP connect phase. Once connected, all reads keep their endpoint timeout,
# so long model generations are never cut short by this knob — only dead/unreachable peers.
HTTP_CONNECT_TIMEOUT = _env_float("HTTP_CONNECT_TIMEOUT", 8.0)

# ---- model-to-model optimizations (all additive, all fail-open) -------------------------
# Circuit breaker: after N consecutive transport failures a brain is bypassed for the rest
# of the session instead of being re-tried (and re-timing-out) on every single round.
BREAKER_THRESHOLD = _env_int("BREAKER_THRESHOLD", 3, "CIRCUIT_BREAK_THRESHOLD")
BREAKER_COOLDOWN = _env_float("BREAKER_COOLDOWN", 180.0, "CIRCUIT_BREAK_COOLDOWN")
# Session-scoped review cache: an identical output+instruction that Qwen already approved
# "ok" is not re-reviewed. The digest is content-addressed (output bytes), so a changed
# result always goes back through the reviewer.
REVIEW_CACHE = _env_bool("REVIEW_CACHE", True)
# Gemma-driven memory compaction: when history grows past MAX_HISTORY, the CEO itself
# distills what still matters into a small priority-memory block instead of plain trimming.
COMPACT_MEMORY = _env_bool("COMPACT_MEMORY", True)
COMPACT_MEMORY_MAX = _env_int("COMPACT_MEMORY_MAX", 700)
COMPACT_MEMORY_KEEP = _env_int("COMPACT_MEMORY_KEEP", 6)
COMPACT_CONTEXT_MAX = _env_int("COMPACT_CONTEXT_MAX", 9000)
# Needle drafts one concrete retry call when the verifier says the work is not satisfied.
DRAFT_FIX = _env_bool("DRAFT_FIX", True)
# Compaction is deterministic (the CLI measures the window; Gemma only writes the note).
# This extra switch lets the same machinery run *mid-task* between refinement rounds, which
# the context-compression literature finds is where the real savings live.
COMPACT_MID_TASK = _env_bool("COMPACT_MID_TASK", False)

# Session-scoped state. Cleared on /reset or a fresh process; never touches config or disk.
_session_memory: List[str] = []
_review_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_circuit: Dict[str, Dict[str, Any]] = {}
# The review pipeline runs on a background thread while the main thread keeps working, so
# every read-modify-write of this shared state takes the lock. It is only ever held for a
# dict lookup/store, so it can never serialize a model call.
_STATE_LOCK = threading.Lock()
# ~3 characters per token is a deliberately conservative estimate for code + English mix.
CHARS_PER_TOKEN = _env_float("CHARS_PER_TOKEN", 3.0)

# Only these have no side effects, so a consecutive run of them can never depend on
# each other's results and is therefore safe to execute concurrently.
READ_ONLY_KINDS = frozenset({
    "read_file", "list_files", "search_files", "web_search", "fetch_url", "memory_retrieve",
})


# ============================================================
# NEEDLE BRAIN (the local one)
# ============================================================
# Third brain, and the only one that runs on THIS device. Needle 2 is a 45M-parameter,
# 2-bit tool-calling model shipped as a single ~14MB binary in ~28MB of RAM: grammar
# constrained JSON out, calibrated confidence on every call, no network at inference.
#
# Division of labour across the three brains:
#   Needle  -> local. Decides and does: tool selection, execution planning, verification,
#              extraction, scope. It never writes prose.
#   Gemma   -> remote Ollama. The CEO: reasoning, diagnosis, and the user-facing answer.
#   Qwen    -> remote Ollama. Reviews failures and tells Gemma what to fix.
#
# Needle emits tool calls, never free text, so it can never answer the user directly.
# Every path below therefore fails OPEN: any error, timeout, low confidence or missing
# slot falls through to exactly the v4 behaviour.
NEEDLE_ROLES = ("refuser", "router", "planner", "verifier", "extractor")
NEEDLE_ROLE_BRIEF = {
    "refuser": "is this even a tool task? off-topic -> let Gemma just answer",
    "router": "which tool kind does this request need, when Gemma proposes none",
    "planner": "completes Gemma's script into concrete tool arguments",
    "verifier": "did the executed work actually satisfy the request",
    "extractor": "shrink oversized tool results into structured fields",
}
NEEDLE_HOST = os.getenv("NEEDLE_HOST", "127.0.0.1:8000")
# The shim lazily spawns one stock engine per role's (system, tools) key, so each role's
# FIRST call pays process startup + cold model load on the phone. A 30s read timeout raced
# that cold start and read-timeouts were misread as outages (socket.timeout ->
# NeedleUnavailable -> breaker). 120s matches the shim's own per-request default.
NEEDLE_TIMEOUT = float(os.getenv("NEEDLE_TIMEOUT", "120"))
NEEDLE_CONFIDENCE = float(os.getenv("NEEDLE_CONFIDENCE", "0.60"))
NEEDLE_MAX_TOKENS = int(os.getenv("NEEDLE_MAX_TOKENS", "512"))
# Tool results at least this big are handed to the extractor before they touch Gemma's
# context. Below it, extraction would cost more than it saves.
NEEDLE_EXTRACT_MIN = int(os.getenv("NEEDLE_EXTRACT_MIN", "4000"))
NEEDLE_EXTRACT_MAX = int(os.getenv("NEEDLE_EXTRACT_MAX", "1400"))
NEEDLE_MASTER = os.getenv("NEEDLE", "0").lower() not in {"0", "false", "no"}
# Per-stage switches. Off = that stage is skipped entirely, not attempted and ignored.
NEEDLE_PLANNER = os.getenv("NEEDLE_PLANNER", "1").lower() not in {"0", "false", "no"}
NEEDLE_ROUTER = os.getenv("NEEDLE_ROUTER", "1").lower() not in {"0", "false", "no"}
NEEDLE_REFUSER = os.getenv("NEEDLE_REFUSER", "1").lower() not in {"0", "false", "no"}
NEEDLE_VERIFIER = os.getenv("NEEDLE_VERIFIER", "1").lower() not in {"0", "false", "no"}
NEEDLE_EXTRACTOR = os.getenv("NEEDLE_EXTRACTOR", "1").lower() not in {"0", "false", "no"}

# Qwen's job: read EVERY tool output — success or failure — and hand Gemma a status,
# a diagnosis and a concrete next step. Off = Gemma reads raw results, as it did in v4.
QWEN_REVIEW = os.getenv("QWEN_REVIEW", "1").lower() not in {"0", "false", "no"}

NEEDLE_SLOT_DEFAULT = {
    "url": "",
    "key": "",
    "model": "needle-2",
    "threshold": None,     # None -> fall back to NEEDLE_CONFIDENCE
    "enabled": True,
    "format": "openai",    # openai | native | auto
}


def _load_needle_slots() -> Dict[str, Dict[str, Any]]:
    """Build the five role slots: config.json wins over defaults, env wins over both."""
    raw = _ENDPOINT_CONFIG.get("needle")
    slots: Dict[str, Dict[str, Any]] = {}
    for role in NEEDLE_ROLES:
        slot = dict(NEEDLE_SLOT_DEFAULT)
        if isinstance(raw, dict) and isinstance(raw.get(role), dict):
            slot.update({k: v for k, v in raw[role].items() if k in slot})
        env = role.upper()
        slot["url"] = str(os.getenv(f"NEEDLE_{env}_URL") or slot["url"] or "").strip()
        slot["key"] = str(os.getenv(f"NEEDLE_{env}_KEY") or slot["key"] or "")
        slot["model"] = str(os.getenv(f"NEEDLE_{env}_MODEL") or slot["model"] or "needle-2")
        slot["format"] = str(os.getenv(f"NEEDLE_{env}_FORMAT") or slot["format"] or "openai")
        thr = os.getenv(f"NEEDLE_{env}_THRESHOLD")
        if thr:
            try:
                slot["threshold"] = float(thr)
            except ValueError:
                pass
        if NEEDLE_MASTER:
            # NEEDLE=1 lights up all five on the local default endpoint in one shot.
            slot["url"] = slot["url"] or NEEDLE_HOST
            slot["enabled"] = True
        if not slot["url"]:
            slot["enabled"] = False
        slots[role] = slot
    return slots


NEEDLE_SLOTS = _load_needle_slots()


def needle_threshold(slot: Dict[str, Any]) -> float:
    try:
        return float(slot.get("threshold"))
    except (TypeError, ValueError):
        return NEEDLE_CONFIDENCE


def needle_active_roles(slots: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    src = slots if slots is not None else NEEDLE_SLOTS
    active: List[str] = []
    for role in NEEDLE_ROLES:
        slot = src.get(role)
        # A hand-edited config.json can put a null/list where a slot object belongs; that
        # must not take down role enumeration (and with it LOCAL_EXEC and the whole start).
        if isinstance(slot, dict) and slot.get("enabled") and slot.get("url"):
            active.append(role)
    return active


def save_needle_slots(slots: Dict[str, Dict[str, Any]]) -> None:
    """Persist the five slots into config.json, preserving every unrelated key."""
    cfg = _load_endpoint_config()
    cfg["needle"] = {r: slots.get(r) for r in NEEDLE_ROLES}
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = CONFIG_FILE + ".tmp"
    data = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CONFIG_FILE)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


# Where Gemma's script actually runs. On = this device, straight through dispatch_worker()
# with no Tailscale round trip; off = the remote Tool Wielder, exactly as in v4. It follows
# the Needle config by default so pointing a role at a local endpoint is what makes this
# box the executioner, and NEEDLE_LOCAL_EXEC pins it either way.
_LOCAL_EXEC_ENV = os.getenv("NEEDLE_LOCAL_EXEC")


def compute_local_exec(slots: Dict[str, Dict[str, Any]]) -> bool:
    """Should Gemma's script run here? Needs an explicit opt-in, or an active role."""
    if _LOCAL_EXEC_ENV is not None:
        return _LOCAL_EXEC_ENV.lower() not in {"0", "false", "no"}
    return bool(needle_active_roles(slots))


LOCAL_EXEC = compute_local_exec(NEEDLE_SLOTS)


# ============================================================
# TRUSTED ENDPOINT / CLI OVERRIDES
# ============================================================

def validate_ipv4(value: str) -> str:
    value = str(value).strip()
    if value.startswith("http://") or value.startswith("https://"):
        raise ValueError("Enter the IPv4 address only, without http:// or a port.")
    try:
        ip = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError("Invalid IPv4 address.") from exc
    if ip.version != 4:
        raise ValueError("Only IPv4 addresses are supported by --ip.")
    return str(ip)


def save_endpoint_config(main_host: Optional[str] = None, tool_host: Optional[str] = None) -> None:
    cfg = _load_endpoint_config()
    if main_host is not None:
        cfg["main_host"] = main_host
    if tool_host is not None:
        cfg["tool_host"] = tool_host
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = CONFIG_FILE + ".tmp"
    data = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CONFIG_FILE)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def configure_ip_interactively() -> None:
    global MAIN_HOST, TOOL_HOST

    println(f"{BOLD}Which server IP do you want to change?{RESET}")
    println("1) Main Brain  (Gemma / Ollama)")
    println("2) Tool Wielder (Qwen / API)")
    choice = input("Select [1/2]: ").strip()
    if choice not in {"1", "2"}:
        raise SystemExit("Invalid choice. Use 1 or 2.")

    label = "Main Brain" if choice == "1" else "Tool Wielder"
    while True:
        raw = input(f"Enter new {label} IPv4: ").strip()
        try:
            ip = validate_ipv4(raw)
            break
        except ValueError as exc:
            println(f"{YELLOW}⚠ {exc}{RESET}")

    if choice == "1":
        MAIN_HOST = f"{ip}:{MAIN_PORT}"
        save_endpoint_config(main_host=MAIN_HOST)
        println(f"{GREEN}✓ Main Brain IP saved: {MAIN_HOST}{RESET}")
    else:
        TOOL_HOST = ip
        save_endpoint_config(tool_host=TOOL_HOST)
        println(f"{GREEN}✓ Tool Wielder IP saved: {TOOL_HOST}:{TOOL_PORT}{RESET}")


def apply_cli_overrides(args: argparse.Namespace) -> None:
    global TOOL_WIELDER_KEY

    if getattr(args, "key", None) is not None:
        if args.key == "-":
            TOOL_WIELDER_KEY = getpass.getpass("Tool Wielder key: ").strip()
        else:
            TOOL_WIELDER_KEY = args.key.strip()
        if not TOOL_WIELDER_KEY:
            raise SystemExit("The Tool Wielder key cannot be empty.")

    if getattr(args, "ip", False):
        configure_ip_interactively()


# ============================================================
# TERMINAL UI
# ============================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"


_print_lock = threading.Lock()


def println(text: str = "") -> None:
    # Serializes output so parallel tool batches cannot interleave half-written lines.
    with _print_lock:
        print(text, flush=True)


def eprintln(text: str) -> None:
    with _print_lock:
        print(f"{RED}{text}{RESET}", file=sys.stderr, flush=True)


def clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---- session circuit breaker + review cache ------------------------------------------

def _breaker_ready(key: str) -> bool:
    """True when the brain named by `key` may be asked again."""
    with _STATE_LOCK:
        state = _circuit.get(key)
        return not state or state.get("until", 0.0) <= time.monotonic()


def _breaker_fail(key: str, reason: Optional[str] = None) -> None:
    should_warn = False
    detail = ""
    with _STATE_LOCK:
        state = _circuit.setdefault(key, {"streak": 0, "until": 0.0, "noticed": False})
        state["streak"] = int(state.get("streak", 0)) + 1
        if reason:
            state["reason"] = reason
        if state["streak"] >= BREAKER_THRESHOLD and state.get("until", 0.0) <= time.monotonic():
            state["until"] = time.monotonic() + BREAKER_COOLDOWN
            state["streak"] = 0
            if not state.get("noticed"):
                state["noticed"] = True
                should_warn = True
                detail = str(state.get("reason") or "").strip()
    if should_warn:
        suffix = f" (last error: {detail})" if detail else ""
        eprintln(f"{YELLOW}⚠ {key} degraded (repeated failures){suffix} — "
                 f"bypassing it for {BREAKER_COOLDOWN:.0f}s, then retrying automatically{RESET}")


def _breaker_ok(key: str) -> None:
    with _STATE_LOCK:
        state = _circuit.get(key)
        if state:
            state["streak"] = 0
            state["until"] = 0.0


def _breaker_reset() -> None:
    with _STATE_LOCK:
        _circuit.clear()


def _review_digest(result: Dict[str, Any]) -> str:
    """Content-addressed fingerprint of a tool result, ignoring run-specific noise.

    duration_ms and task_id change every run; the digest must not, or identical outputs
    would never cache-hit. Only the information the reviewer actually judges counts — and
    it must cover ALL of it: the old version hashed only the first 800 characters and the
    first six list items, so a change past those boundaries still produced the same digest
    and could reuse a stale "ok". Tool results are already clipped to MAX_TOOL_RESULT, so
    hashing them whole is bounded.
    """
    fields: Dict[str, Any] = {}
    for key in ("ok", "kind", "path", "exit_code", "timed_out", "error"):
        fields[key] = result.get(key)
    for key in ("content", "stdout", "stderr", "text"):
        value = result.get(key)
        if isinstance(value, str):
            fields[key] = value
    for key in ("results", "entries", "memories", "matches"):
        value = result.get(key)
        if isinstance(value, list):
            fields[key] = value
    return hashlib.sha256(compact_json(fields).encode("utf-8")).hexdigest()


def _review_cache_key(args: Dict[str, Any], result: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    if not REVIEW_CACHE:
        return None
    label = str(
        args.get("instruction") or args.get("path") or args.get("query")
        or args.get("url") or ""
    ).strip()
    kind = str(args.get("kind") or result.get("kind") or "unknown").strip().lower()
    if not label:
        return None
    return (kind, label, _review_digest(result))


def _review_cache_hit(args: Dict[str, Any], result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = _review_cache_key(args, result)
    if key is None:
        return None
    with _STATE_LOCK:
        cached = _review_cache.get(key)
        return dict(cached) if cached is not None else None


def _review_cache_store(args: Dict[str, Any], result: Dict[str, Any], review: Dict[str, Any]) -> None:
    if not isinstance(review, dict) or review.get("status") != "ok":
        # Only tide-approved verdicts are deterministic enough to reuse.
        return
    key = _review_cache_key(args, result)
    if key is not None:
        with _STATE_LOCK:
            _review_cache[key] = dict(review)


def _review_cache_reset() -> None:
    with _STATE_LOCK:
        _review_cache.clear()


_REVIEW_CAPABILITY_MARKERS = (
    "unknown action", "unknown tool kind", "no such action",
    "not supported", "unsupported", "not found",
)


def _review_capability_error(body: Any) -> bool:
    """True when a worker's HTTP error says the 'review' action itself is unknown.

    A worker that answers at all is *reachable*, so an "unknown action" / "not found"
    reply is a capability mismatch (an older Tool Wielder with no review branch), not an
    outage. Counting it as a transport failure made every round re-warn
    "review:qwen degraded" about a feature that could never answer.
    """
    low = str(body or "").lower()
    return any(marker in low for marker in _REVIEW_CAPABILITY_MARKERS)


def _disable_review(reason: str) -> None:
    """Turn the reviewer off for the session after a capability error, once, clearly.

    This is the documented fail-open: results simply go to the CEO unreviewed, exactly
    as a run with QWEN_REVIEW=0 would behave.
    """
    global QWEN_REVIEW
    if not QWEN_REVIEW:
        return
    QWEN_REVIEW = False
    eprintln(f"{YELLOW}⚠ review off — {reason}; results go to the CEO unreviewed{RESET}")


# ============================================================
# GENERIC PERSISTENT HTTP
# ============================================================

class HTTPFailure(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = int(status)
        self.body = body
        super().__init__(f"HTTP {self.status}: {clip(body, 1200)}")


# Transport-level failures worth tripping the circuit breaker for. Everything else (a bug
# in our own parsing, a TypeError, a KeyError) must NOT masquerade as a dead reviewer and
# silently disable Qwen for the session.
_TRANSPORT_ERRORS = (OSError, HTTPFailure)


def _connect_timeout_class(secure: bool, connect_timeout: float) -> Callable[..., HTTPConnection]:
    """HTTP(S) connection factory whose TCP connect fails fast while reads stay generous.

    A wedged peer used to eat the FULL read timeout just to fail a TCP handshake — with a
    300s MAIN_TIMEOUT, one unreachable host stalls the whole turn. The connect cap is pure
    fail-fast: once the socket is up, every read keeps the longer timeout the caller chose,
    so a model that is genuinely generating still gets all the time it needs.
    """
    base = HTTPSConnection if secure else HTTPConnection

    class _FastConnect(base):  # type: ignore[valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._fast_connect_timeout = connect_timeout
            # CRITICAL: HTTPConnection.__init__ installs a *plain* create_connection as an
            # INSTANCE attribute, which shadows any subclass method of the same name. The
            # original class-level override therefore never ran and every connect still ate
            # the full read timeout. We must rebind the instance attribute after super().
            self._create_connection = self._fast_create_connection

        def _fast_create_connection(self, address: Tuple[str, int],
                                    timeout: Optional[float] = None,
                                    source_address: Optional[Tuple[str, int]] = None) -> Any:
            return socket.create_connection(
                address, timeout=self._fast_connect_timeout, source_address=source_address
            )

        def connect(self) -> None:
            try:
                super().connect()
            except BaseException:
                # Never leave a half-open socket behind after a failed fast connect.
                sock = getattr(self, "sock", None)
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                    self.sock = None
                raise
            # create_connection leaves the socket at the tiny connect cap. Restore the
            # caller's generous read timeout, otherwise generations get cut short at the
            # connect timeout — which is exactly what this whole class exists to prevent.
            sock = getattr(self, "sock", None)
            if sock is not None:
                try:
                    sock.settimeout(self.timeout)
                except OSError:
                    pass

    return _FastConnect


def _split_host_port(host: str, default_port: int) -> Tuple[str, int]:
    """Split `host`, `host:port`, `[::1]`, `[::1]:8000` or a bare IPv6 literal.

    The old `":" in host` check mangled every IPv6 form: `[::1]:8000` kept the brackets
    (breaking getaddrinfo) and a bare `::1` was read as host="" port=1.
    """
    host = host.strip()
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            name = host[1:end]
            rest = host[end + 1:]
            if rest.startswith(":") and rest[1:].isdigit():
                return name, int(rest[1:])
            return name, default_port
    if host.count(":") > 1:
        return host, default_port
    if ":" in host:
        name, raw_port = host.rsplit(":", 1)
        if raw_port.isdigit():
            return name, int(raw_port)
    return host, default_port


class PersistentHTTP:
    """Single reusable HTTP/1.1 connection with one reconnect retry."""

    def __init__(self, host: str, timeout: float, secure: bool = False,
                 connect_timeout: Optional[float] = None):
        self.host, self.port = _split_host_port(host, 443 if secure else 80)
        self.timeout = timeout
        self.secure = bool(secure)
        self.connect_timeout = float(connect_timeout) if connect_timeout is not None else HTTP_CONNECT_TIMEOUT
        self._conn_cls = _connect_timeout_class(self.secure, self.connect_timeout)
        self._conn: Optional[HTTPConnection] = None
        self._lock = threading.Lock()

    def _connect(self, timeout: Optional[float] = None) -> HTTPConnection:
        return self._conn_cls(self.host, self.port, timeout=timeout or self.timeout)

    def _new(self) -> HTTPConnection:
        conn = self._connect()
        self._conn = conn
        return conn

    def close(self) -> None:
        # Best-effort shutdown: never block behind an in-flight request that is itself
        # waiting out a connect timeout (prewarm can be wedged on an unreachable host).
        # If the lock is busy, the process is exiting anyway and the socket dies with it.
        if not self._lock.acquire(timeout=0.5):
            return
        try:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        finally:
            self._lock.release()

    def request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        headers = headers or {}
        last_exc: Optional[BaseException] = None

        with self._lock:
            for attempt in range(2):
                conn = self._conn or self._new()
                try:
                    conn.request(method, path, body=body, headers=headers)
                    response = conn.getresponse()
                    data = response.read()
                    if response.status >= 400:
                        raise HTTPFailure(response.status, data.decode(errors="replace"))
                    return data
                except (ConnectionError, BrokenPipeError, OSError, socket.timeout) as exc:
                    last_exc = exc
                    try:
                        conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if attempt == 1:
                        raise
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    raise

        raise RuntimeError(str(last_exc))

    def open_stream(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[HTTPConnection, Any]:
        """Open a dedicated connection for an incremental (streaming) response.

        A fresh socket is used on purpose: sharing `self._conn` would let another thread
        issue a request while we are still draining this one, corrupting both. The caller
        owns the connection and must close it.
        """
        headers = headers or {}
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            conn = self._connect()
            try:
                conn.request(method, path, body=body, headers=headers)
                return conn, conn.getresponse()
            except (ConnectionError, BrokenPipeError, OSError, socket.timeout) as exc:
                last_exc = exc
                try:
                    conn.close()
                except Exception:
                    pass
                if attempt == 1:
                    raise
            except Exception:
                # BadStatusLine/IncompleteRead and friends are not OSError, so the transport
                # branch misses them and the dedicated socket used to leak. Close it here.
                try:
                    conn.close()
                except Exception:
                    pass
                raise
        raise RuntimeError(str(last_exc))


# ============================================================
# OLLAMA CLIENTS
# ============================================================

class MainOllama:
    def __init__(self):
        self.http = PersistentHTTP(MAIN_HOST, MAIN_TIMEOUT)
        self._generation_lock = threading.Lock()

    def close(self) -> None:
        self.http.close()

    def tags(self) -> Dict[str, Any]:
        return json.loads(
            self.http.request(
                "GET",
                "/api/tags",
                headers={"Accept": "application/json", "Connection": "keep-alive"},
            )
        )

    def ps(self) -> Dict[str, Any]:
        return json.loads(
            self.http.request(
                "GET",
                "/api/ps",
                headers={"Accept": "application/json", "Connection": "keep-alive"},
            )
        )

    def warm(self) -> None:
        body = compact_json({
            "model": MAIN_MODEL,
            "prompt": " ",
            "stream": False,
            "think": False,
            "keep_alive": MAIN_KEEP_ALIVE,
            "options": {"num_ctx": MAIN_CTX, "num_predict": 1},
        }).encode()
        with self._generation_lock:
            self.http.request(
                "POST",
                "/api/generate",
                body,
                {"Content-Type": "application/json", "Accept": "application/json", "Connection": "keep-alive"},
            )

    def _chat_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_tools: bool,
        num_predict: Optional[int],
        num_ctx: Optional[int],
        temperature: float,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": MAIN_MODEL,
            "messages": messages,
            "stream": stream,
            "think": False,
            "keep_alive": MAIN_KEEP_ALIVE,
            "options": {
                "num_ctx": int(num_ctx or MAIN_CTX),
                "num_predict": int(num_predict or MAIN_MAX_OUTPUT),
                "temperature": float(temperature),
                "top_p": 0.9,
            },
        }
        if use_tools:
            payload["tools"] = MAIN_TOOLS
        return payload

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_tools: bool = True,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload = self._chat_payload(
            messages,
            use_tools=use_tools,
            num_predict=num_predict,
            num_ctx=num_ctx,
            temperature=MAIN_TEMPERATURE if temperature is None else temperature,
            stream=False,
        )
        body = compact_json(payload).encode()
        with self._generation_lock:
            raw = self.http.request(
                "POST",
                "/api/chat",
                body,
                {"Content-Type": "application/json", "Accept": "application/json", "Connection": "keep-alive"},
            )
        return json.loads(raw)

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_tools: bool = False,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
        temperature: Optional[float] = None,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Stream a response, pushing each content fragment to `on_delta` as it arrives.

        Returns the same shape as `chat()` so callers can treat them interchangeably.
        Only used when the request carries no tools, because a model cannot emit tool_calls
        against a payload that did not ask for them — so nothing printed can be invalidated.
        """
        payload = self._chat_payload(
            messages,
            use_tools=use_tools,
            num_predict=num_predict,
            num_ctx=num_ctx,
            temperature=MAIN_TEMPERATURE if temperature is None else temperature,
            stream=True,
        )
        body = compact_json(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
        }

        parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        final: Dict[str, Any] = {}
        saw_done = False

        with self._generation_lock:
            conn, response = self.http.open_stream("POST", "/api/chat", body, headers)
            try:
                read = getattr(response, "read1", None) or response.read
                buf = b""
                while True:
                    chunk = read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw_line, buf = buf.split(b"\n", 1)
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            event = json.loads(raw_line)
                        except ValueError:
                            continue
                        if isinstance(event, dict) and event.get("error"):
                            raise RuntimeError(str(event["error"]))
                        msg = (event or {}).get("message") or {}
                        piece = msg.get("content") or ""
                        if piece:
                            parts.append(piece)
                            if on_delta is not None:
                                on_delta(piece)
                        extra = msg.get("tool_calls")
                        if extra:
                            tool_calls.extend(extra)
                        if (event or {}).get("done"):
                            final = event
                            saw_done = True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        # A stream that just stops — socket closed, proxy timeout, Ollama crash — used to be
        # indistinguishable from a completed answer. Refusing to call that "done" is the
        # whole point: the caller then treats it as the transport failure it is.
        if not saw_done:
            raise RuntimeError("stream ended before completion (connection dropped)")

        content = "".join(parts)
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "model": final.get("model", MAIN_MODEL),
            "message": message,
            "done": True,
            "done_reason": final.get("done_reason", "stop"),
            "eval_count": final.get("eval_count"),
            "total_duration": final.get("total_duration"),
        }


class QwenOllama:
    def __init__(self):
        self.http = PersistentHTTP(QWEN_HOST, QWEN_TIMEOUT)
        self._lock = threading.Lock()

    def close(self) -> None:
        self.http.close()

    def tags(self) -> Dict[str, Any]:
        return json.loads(
            self.http.request(
                "GET",
                "/api/tags",
                headers={"Accept": "application/json", "Connection": "keep-alive"},
            )
        )

    def warm(self) -> None:
        body = compact_json({
            "model": QWEN_MODEL,
            "prompt": " ",
            "stream": False,
            "think": False,
            "keep_alive": QWEN_KEEP_ALIVE,
            "options": {"num_ctx": QWEN_CTX, "num_predict": 1},
        }).encode()
        with self._lock:
            self.http.request(
                "POST",
                "/api/generate",
                body,
                {"Content-Type": "application/json", "Accept": "application/json", "Connection": "keep-alive"},
            )

    def json_chat(self, system: str, user: str, max_predict: int) -> Dict[str, Any]:
        body = compact_json({
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": QWEN_KEEP_ALIVE,
            "format": "json",
            "options": {
                "num_ctx": QWEN_CTX,
                "num_predict": max_predict,
                "temperature": 0.0,
                "top_p": 0.9,
            },
        }).encode()
        with self._lock:
            raw = self.http.request(
                "POST",
                "/api/chat",
                body,
                {"Content-Type": "application/json", "Accept": "application/json", "Connection": "keep-alive"},
            )
        result = json.loads(raw)
        content = ((result.get("message") or {}).get("content") or "").strip()
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(content[start:end + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    pass
        return {}


# ============================================================
# NEEDLE BRAIN — the local executioner
# ============================================================

def _split_needle_url(url: str) -> Tuple[str, bool, str]:
    """'https://host:8000/v1' -> ('host:8000', True, '/v1')."""
    raw = str(url).strip()
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if not parsed.hostname:
        raise ValueError(f"Needle endpoint has no host: {url!r}")
    secure = parsed.scheme == "https"
    port = parsed.port or (443 if secure else 80)
    return f"{parsed.hostname}:{port}", secure, (parsed.path or "").rstrip("/")


def _flat_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """OpenAI-wrapped tool specs -> Needle's flat {name, description, parameters}."""
    flat: List[Dict[str, Any]] = []
    for spec in tools or []:
        if not isinstance(spec, dict):
            continue
        if spec.get("type") == "function" and isinstance(spec.get("function"), dict):
            fn = spec["function"]
            flat.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object"},
            })
        elif spec.get("name"):
            flat.append(spec)
    return flat


def _openai_tool(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


def _normalise_calls(raw: Any) -> List[Dict[str, Any]]:
    """Whatever shape the endpoint returned -> [{name, arguments:dict}, ...]."""
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        args = item.get("arguments")
        fn = item.get("function")
        if isinstance(fn, dict):
            name = name or fn.get("name")
            if args is None:
                args = fn.get("arguments")
        if not name:
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                # The grammar is supposed to make this impossible; keep it visible anyway.
                args = {"_raw": args}
        elif args is None:
            args = {}
        elif not isinstance(args, dict):
            args = {"value": args}
        out.append({"name": str(name), "arguments": args})
    return out


# Schemas for the non-execution roles. All of them are closed objects so Needle's
# byte-level grammar admits exactly one shape: the output cannot come back malformed.
NEEDLE_SCOPE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tool_task", "reason"],
    "properties": {
        "tool_task": {"type": "boolean",
                      "description": "True only if the request needs a tool to answer."},
        "reason": {"type": "string", "max_length": 300},
    },
}

NEEDLE_VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["satisfied", "gap"],
    "properties": {
        "satisfied": {"type": "boolean",
                      "description": "True if the produced output fulfils the instruction."},
        "gap": {"type": "string", "max_length": 400,
                "description": "What is still missing. Empty string when satisfied."},
    },
}

NEEDLE_EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "facts"],
    "properties": {
        "summary": {"type": "string", "max_length": 1200,
                    "description": "Tight summary of what the output actually says."},
        "facts": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "max_length": 200},
            "description": "Concrete values: paths, numbers, names, statuses, error text.",
        },
    },
}


class NeedleBrain:
    """One local Needle 2 endpoint bound to one role.

    Needle is an executioner, not an author: it never produces prose, only grammar
    constrained JSON. `complete()` therefore returns *calls*, and when it returns
    nothing the caller escalates straight to Gemma. No method here raises into the
    agent loop — a dead Needle must look exactly like no Needle at all.
    """

    def __init__(self, role: str, slot: Dict[str, Any]):
        self.role = role
        hostport, secure, base = _split_needle_url(str(slot.get("url") or ""))
        self.key = str(slot.get("key") or "")
        self.model = str(slot.get("model") or "needle-2")
        self.threshold = needle_threshold(slot)
        self.timeout = float(slot.get("timeout") or NEEDLE_TIMEOUT)
        requested = str(slot.get("format") or "openai").lower()
        # None = not yet decided; auto-detection picks it up from the first response.
        self._fmt: Optional[str] = None if requested in ("auto", "") else requested
        self._auto = requested in ("auto", "")
        if base.endswith("/v1"):
            self._base = base
            self._root = base[:-len("/v1")]
        elif base:
            self._base = base + "/v1"
            self._root = base
        else:
            self._base = "/v1"
            self._root = ""
        self.http = PersistentHTTP(hostport, self.timeout, secure=secure)

    # ---- plumbing -------------------------------------------------------
    def close(self) -> None:
        self.http.close()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def health(self) -> bool:
        try:
            # needle-openai exposes /health at the mount root, not under /v1, so this
            # has to follow the endpoint's path prefix instead of assuming "/"
            self.http.request("GET", f"{self._root}/health",
                              headers={"Accept": "application/json"})
            return True
        except Exception:
            return False

    # ---- calls ----------------------------------------------------------
    def _post_openai(self, system: str, user: str,
                     tools: Optional[List[Dict[str, Any]]],
                     schema: Optional[Dict[str, Any]],
                     max_tokens: int) -> Dict[str, Any]:
        spec = tools
        if schema is not None:
            spec = [_openai_tool("submit", "Submit the requested structured result.", schema)]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": int(max_tokens),
        }
        if spec:
            payload["tools"] = spec
        data = json.loads(self.http.request(
            "POST", f"{self._base}/chat/completions",
            compact_json(payload).encode(), self._headers(),
        ))
        choice = ((data.get("choices") or [{}])[0]) if isinstance(data, dict) else {}
        message = choice.get("message") or {}
        calls = _normalise_calls(message.get("tool_calls"))
        confidence = None
        extras = data.get("x_needle")
        if isinstance(extras, dict):
            confidence = extras.get("confidence")
        if confidence is None and isinstance(data, dict):
            confidence = data.get("confidence")
        return {
            "ok": True,
            "calls": calls,
            "confidence": _coerce_confidence(confidence),
            "reasoning": str(message.get("content") or ""),
            "raw": data,
        }

    def _post_native(self, system: str, user: str,
                     tools: Optional[List[Dict[str, Any]]],
                     schema: Optional[Dict[str, Any]],
                     max_tokens: int) -> Dict[str, Any]:
        spec = tools
        if schema is not None:
            spec = [{
                "name": "submit",
                "description": "Submit the requested structured result.",
                "parameters": schema,
            }]
        payload: Dict[str, Any] = {
            "query": user,
            "system": system,
            "max_new_tokens": int(max_tokens),
        }
        if spec:
            payload["tools"] = _flat_tools(spec)
        data = json.loads(self.http.request(
            "POST", f"{self._base}/needle/complete",
            compact_json(payload).encode(), self._headers(),
        ))
        if not isinstance(data, dict):
            raise ValueError("native Needle response was not an object")
        return {
            "ok": True,
            "calls": _normalise_calls(data.get("function_calls")),
            "confidence": _coerce_confidence(data.get("confidence")),
            "reasoning": str(data.get("reasoning") or ""),
            "raw": data,
        }

    def complete(self, *, system: str, user: str,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 schema: Optional[Dict[str, Any]] = None,
                 max_tokens: int = NEEDLE_MAX_TOKENS,
                 strict: bool = False) -> Optional[Dict[str, Any]]:
        """One Needle turn. Returns a normalised result, or None to escalate to Gemma.

        `strict=True` re-raises transport-level failures (connect/timeout/broken pipe)
        so the caller's circuit breaker can measure availability. It still returns None
        for endpoint-level refusals (404/405/format mismatch), which are config issues,
        not outages.
        """
        order = [self._fmt] if self._fmt else (
            ["openai", "native"] if self._auto else [self._fmt or "openai"])
        last_exc: Optional[BaseException] = None
        for fmt in order:
            try:
                if fmt == "native":
                    result = self._post_native(system, user, tools, schema, max_tokens)
                else:
                    result = self._post_openai(system, user, tools, schema, max_tokens)
            except HTTPFailure as exc:
                # 404/405 on the OpenAI route means this endpoint only speaks native.
                if fmt == "openai" and self._auto and exc.status in (404, 405) and len(order) > 1:
                    continue
                # A 5xx is the endpoint being down, not a config mismatch — under strict mode
                # it must surface so the router's breaker can see the outage.
                if strict and exc.status >= 500:
                    raise NeedleUnavailable(f"HTTP {exc.status}") from exc
                return None
            except (ConnectionError, BrokenPipeError, OSError, socket.timeout) as exc:
                last_exc = exc
                if strict:
                    raise NeedleUnavailable(f"{type(exc).__name__}: {exc}") from exc
                return None
            except Exception:
                return None
            self._fmt = fmt
            return result
        if last_exc is not None and strict:
            raise NeedleUnavailable(f"{type(last_exc).__name__}: {last_exc}")
        return None


class NeedleUnavailable(RuntimeError):
    """A transport-level failure talking to a local Needle endpoint (connect/timeout)."""


def _coerce_confidence(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0:          # some builds report 0-100
        number /= 100.0
    return max(0.0, min(number, 1.0))


class NeedleRouter:
    """The five local roles. Every method fails open: no active role is not an error."""

    def __init__(self, slots: Optional[Dict[str, Dict[str, Any]]] = None):
        self.slots = slots if slots is not None else NEEDLE_SLOTS
        self.clients: Dict[str, NeedleBrain] = {}
        for role in needle_active_roles(self.slots):
            try:
                self.clients[role] = NeedleBrain(role, self.slots[role])
            except Exception:
                pass

    @property
    def active(self) -> List[str]:
        return [role for role in NEEDLE_ROLES if role in self.clients]

    def has(self, role: str) -> bool:
        return role in self.clients

    def threshold(self, role: str) -> float:
        client = self.clients.get(role)
        return client.threshold if client is not None else NEEDLE_CONFIDENCE

    def ask(self, role: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        client = self.clients.get(role)
        if client is None:
            return None
        key = f"needle:{role}"
        if not _breaker_ready(key):
            return None
        try:
            result = client.complete(strict=True, **kwargs)
            _breaker_ok(key)
            return result
        except NeedleUnavailable as exc:
            _breaker_fail(key, str(exc))
            return None
        except Exception:
            # Anything else (bad payload, model-side hiccup) still fails open untouched.
            return None

    def health(self) -> Dict[str, bool]:
        return {role: client.health() for role, client in self.clients.items()}

    def close(self) -> None:
        for client in self.clients.values():
            try:
                client.close()
            except Exception:
                pass


def build_needle_router() -> NeedleRouter:
    """Built once per process. An empty router is normal and costs nothing."""
    return NeedleRouter()


# ============================================================
# GEMMA TOOL SCHEMA
# ============================================================

TOOL_KINDS = (
    "execute",
    "read_file",
    "write_file",
    "append_file",
    "replace_text",
    "delete_file",
    "list_files",
    "search_files",
    "web_search",
    "fetch_url",
    "memory_retrieve",
    "memory_store",
)

TOOL_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "instruction"],
    "properties": {
        "kind": {
            "type": "string",
            "enum": list(TOOL_KINDS),
            "description": "Which tool to run.",
        },
        "instruction": {
            "type": "string",
            "description": (
                "A SHORT one-line label for the task (what this call is doing). "
                "NEVER paste file bodies, code, or shell output here — those go in "
                "the dedicated fields: content, command, query, url."
            ),
        },
        "context": {
            "type": "string",
            "description": "Only the background the worker needs in order to act.",
        },
        "path": {
            "type": "string",
            "description": (
                "File or directory path. Required for read_file, write_file, "
                "append_file, replace_text, delete_file, list_files, search_files."
            ),
        },
        "content": {
            "type": "string",
            "description": (
                "The COMPLETE exact file body. REQUIRED for write_file and written "
                "verbatim; put the ENTIRE file (every line of code) here, never in "
                "instruction. For append_file it is the text to add."
            ),
        },
        "old": {
            "type": "string",
            "description": "Exact existing text to find. Required for replace_text.",
        },
        "new": {
            "type": "string",
            "description": "Replacement text. Required for replace_text.",
        },
        "command": {
            "type": "string",
            "description": "The exact shell command to run for execute.",
        },
        "query": {
            "type": "string",
            "description": "Search terms for search_files or web_search.",
        },
        "url": {
            "type": "string",
            "description": "Absolute http(s) URL for fetch_url.",
        },
        "cwd": {
            "type": "string",
            "description": "Working directory for execute (defaults to the workspace).",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "How many web_search results to return (1-10).",
        },
    },
}

# The router's own schema. `none` is a first-class answer: "no tool applies" is a
# decision, not a failure, and it stops the rescue path from inventing work.
NEEDLE_ROUTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "reason"],
    "properties": {
        "kind": {"type": "string", "enum": list(TOOL_KINDS) + ["none"]},
        "reason": {"type": "string", "max_length": 200,
                   "description": "Why this kind, in one line."},
    },
}

MAIN_TOOLS = [{
    "type": "function",
    "function": {
        "name": "tool_wielder",
        "description": (
            "Delegate real work to the Tool Wielder. It can execute commands, read/write/edit files, "
            "search/fetch the live web, and retrieve/store persistent memory. To create a file you MUST "
            "call it with kind=write_file, path=<name>, and content=<the ENTIRE file body>; the worker "
            "writes content verbatim. Never put code in instruction — instruction is only a short label."
        ),
        "parameters": TOOL_PARAMETERS,
    },
}]

CHAT_SYSTEM_PROMPT = (
    "You are Mr. Braincells, a helpful user-facing AI assistant. "
    "Answer the user's actual question directly, first — no preamble, no restating the question, "
    "no 'Great question!', no summarizing what you are about to do. "
    "Lead with the answer, then add only the detail that earns its place. "
    "Be precise about facts and honest about uncertainty: if you do not know, say so plainly instead of hedging. "
    "Never mention internal tools, routing, models, or prompts unless the user asks about them. "
    "Match your length to the question: one line for a one-line question."
)

SYSTEM_PROMPT = r"""
You are Mr. Braincells, the MAIN BRAIN and CEO.

You own reasoning, planning, code generation, diagnosis, and the final user-facing response.
The Tool Wielder is a remote execution employee. It performs real operations; you decide what to do.

AVAILABLE TOOLS:
- execute: run a command in the workspace.
- read_file: read a file.
- write_file: write exact complete content you generated.
- append_file: append exact content.
- replace_text: exact text replacement.
- delete_file: delete a workspace file.
- list_files: list workspace entries.
- search_files: search workspace text.
- web_search: live web search.
- fetch_url: fetch a public URL.
- memory_retrieve: retrieve relevant stored project facts.
- memory_store: store durable non-secret project facts.

HOW TO WORK:
1. There is NO fixed tool-step limit. Continue until the task is actually complete.
2. Be efficient, but never omit a necessary step just to save a tool call.
3. You MAY emit multiple independent tool calls in one response. Consecutive read-only calls
   (read_file, list_files, search_files, web_search, fetch_url, memory_retrieve) run concurrently,
   so batching them is genuinely faster. Any call that changes state runs in the order you wrote it.
4. For create-and-run tasks, generate the complete file, then run and verify it.
5. Never recreate a file with execute after write_file. Never repeat an identical successful request
   unless verification shows a reason.
6. Combine dependent verification into one command when practical, e.g.
   `python3 -m py_compile file.py && python3 file.py`.
7. For write_file, provide the COMPLETE exact file content. The worker writes it verbatim; it must not
   invent, fill in, or truncate missing code. A partial file is a failed task.
8. After a successful operation, treat the returned result as ground truth unless the task itself
   requires stronger verification.
9. When an operation fails, read the EXACT error text, diagnose the underlying cause, fix it, and
   retry. Never repeat the same failing call unchanged.
10. For fresh/current questions or explicit web-search requests, use web_search and base current
    claims on the returned results; fetch source pages when more detail is needed.
11. Use memory only when relevant; never store secrets, credentials, or tokens.
12. Keep prompts compact without stripping information needed to finish the task.
13. Every tool result is authoritative input for your next decision. Do not pretend you did not
    receive it, and do not claim an operation succeeded unless its result says ok:true.
14. When the user says "continue", "keep going", or similar, resume the last incomplete task from
    the preserved tool history instead of claiming that prior context is unavailable.

STOPPING:
Call tools only while they still change your understanding or the state of the world. As soon as you
can answer, stop calling tools and answer. Do not re-read a file you just wrote, do not re-run a
command that already succeeded, and do not verify a fact the tool already reported.

FINAL ANSWER (when you are done working):
- Start with the outcome. The user wants what they asked for, not a log of how you got there.
- Report concrete facts: file paths written, commands run, exit codes, test results.
- If something failed or was skipped, say so directly. Never paper over a partial result.
- No filler, no restating the task, no "I have now...", no summarizing your tool usage.
  The tool transcript above already shows your work.
"""



# ============================================================
# MEMORY / WEB STATE
# ============================================================

_memory_lock = threading.Lock()
_web_lock = threading.Lock()
_memory_cache: List[Dict[str, Any]] = []
_memory_mtime = 0.0
_memory_version = 0
_memory_index: List[Tuple[Set[str], str, Dict[str, Any]]] = []
_web_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def ensure_memory_dir() -> None:
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)


def load_memory_cache() -> None:
    global _memory_cache, _memory_mtime, _memory_version
    ensure_memory_dir()
    try:
        mtime = os.path.getmtime(MEMORY_FILE)
    except OSError:
        if _memory_cache:
            _memory_cache = []
            _memory_mtime = 0.0
            _memory_version += 1
        return
    if mtime == _memory_mtime and _memory_cache:
        return
    rows: List[Dict[str, Any]] = []
    with _memory_lock:
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
        except OSError:
            rows = []
        _memory_cache = rows[-10000:]
        _memory_mtime = mtime
        _memory_version += 1


_WORD_RE = re.compile(r"[a-z0-9_./:-]+")
_STOP_WORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "how", "about", "with",
    "from", "that", "this", "have", "has", "does", "did", "please", "find",
    "remember", "memory", "memories", "stored", "retrieve", "recall", "the",
    "and", "for", "you", "your", "are", "was", "were", "any", "some", "into",
    "tell", "show", "give", "want", "need", "there", "their", "been", "get",
})


def _ensure_memory_index() -> None:
    """Tokenize rows once per cache generation instead of on every retrieval."""
    global _memory_index
    load_memory_cache()
    cached_version = getattr(_ensure_memory_index, "_version", -1)
    if cached_version == _memory_version:
        return
    index: List[Tuple[Set[str], str, Dict[str, Any]]] = []
    for row in _memory_cache:
        text = f"{row.get('content', '')} {' '.join(map(str, row.get('tags', [])))}".lower()
        tokens = set(_WORD_RE.findall(text))
        index.append((tokens, " ".join(sorted(tokens)), row))
    _memory_index = index
    setattr(_ensure_memory_index, "_version", _memory_version)


def _memory_terms(instruction: str) -> List[str]:
    terms: List[str] = []
    seen: Set[str] = set()
    for tok in _WORD_RE.findall(str(instruction or "").lower()):
        if tok in _STOP_WORDS or len(tok) < 3 or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= 10:
            break
    return terms


def memory_retrieve(instruction: str, qwen: QwenOllama) -> Dict[str, Any]:
    _ensure_memory_index()
    if not _memory_index:
        return {"ok": True, "kind": "memory_retrieve", "memories": []}

    terms = _memory_terms(instruction)
    if not terms:
        return {"ok": True, "kind": "memory_retrieve", "terms": [], "memories": []}

    n_rows = len(_memory_index)
    # Pass 1: document frequency. Rare terms are far more informative than common ones,
    # so "redis" must outrank "project" — plain substring counts get this exactly backwards.
    df = {t: 0 for t in terms}
    for tokens, norm, _row in _memory_index:
        for t in terms:
            if t in tokens or t in norm:
                df[t] += 1
    idf = {t: math.log(1.0 + (n_rows + 1.0) / (df[t] + 1.0)) for t in terms}

    phrase = " ".join(terms)
    now = time.time()
    # (total score, term-only score, row) — term score is kept separately so the cutoff
    # below is driven by query relevance rather than by age or stored importance.
    scored: List[Tuple[float, float, Dict[str, Any]]] = []

    for tokens, norm, row in _memory_index:
        term_score = 0.0
        matched = 0
        for t in terms:
            if t in tokens:
                term_score += idf[t]
                matched += 1
            elif t in norm:
                term_score += idf[t] * 0.5
                matched += 1
        if not matched:
            continue
        if phrase and phrase in norm:
            term_score += 2.0
        importance = min(float(row.get("importance", 1.0) or 1.0), 5.0)
        age_days = max((now - float(row.get("created_at", now))) / 86400.0, 0.0)
        recency = 1.0 / (1.0 + age_days / 30.0)
        scored.append((term_score + importance * 0.25 + recency * 0.25, term_score, row))

    if not scored:
        return {"ok": True, "kind": "memory_retrieve", "terms": terms, "memories": []}

    scored.sort(key=lambda triple: triple[0], reverse=True)
    # Drop rows that only grazed one incidental term. Without this, IDF weighting alone
    # still lets a lone common-word hit outrank nothing and fill the result list.
    cutoff = max(0.5, scored[0][1] * 0.4)
    kept = [triple for triple in scored if triple[1] >= cutoff]
    return {
        "ok": True,
        "kind": "memory_retrieve",
        "terms": terms,
        "memories": [row for _, _, row in kept[:8]],
    }


def memory_store(instruction: str, context: str, qwen: QwenOllama) -> Dict[str, Any]:
    global _memory_mtime, _memory_version
    prompt = clip(instruction, 3000)
    if context:
        prompt += "\nContext:\n" + clip(context, 2000)

    decision: Dict[str, Any] = {}
    try:
        decision = qwen.json_chat(
            (
                'Return JSON only: {"content":"string","tags":["string"],"importance":1.0}. '
                'Store one durable, useful project fact. Never store passwords, API keys, access tokens, '
                'private credentials, or other secrets. If no durable fact exists, use empty content.'
            ),
            prompt,
            140,
        )
    except Exception:
        # Deterministic fallback means memory still functions when Qwen is temporarily unavailable.
        decision = {"content": clip(instruction, 1800), "tags": [], "importance": 1.0}

    content = str(decision.get("content", "")).strip()
    if not content:
        return {"ok": True, "kind": "memory_store", "stored": False}

    sensitive = re.search(
        r"(?i)(api[_ -]?key|access[_ -]?token|password|secret|private[_ -]?key|bearer\s+token)",
        content,
    )
    if sensitive:
        return {"ok": False, "kind": "memory_store", "stored": False, "error": "Refused to persist a credential-like value."}

    try:
        importance = float(decision.get("importance", 1.0))
    except Exception:
        importance = 1.0

    raw_tags = decision.get("tags")
    # A model that answers `"tags": null` used to crash the whole store with TypeError.
    tags = [str(x)[:60] for x in raw_tags[:10]] if isinstance(raw_tags, list) else []
    obj = {
        "id": f"m{time.time_ns()}",
        "created_at": time.time(),
        "content": content[:2000],
        "tags": tags,
        "importance": max(0.1, min(importance, 5.0)),
    }

    ensure_memory_dir()
    line = compact_json(obj) + "\n"
    with _memory_lock:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        _memory_cache.append(obj)
        if len(_memory_cache) > 10000:
            del _memory_cache[:-10000]
        _memory_version += 1
        try:
            _memory_mtime = os.path.getmtime(MEMORY_FILE)
        except OSError:
            pass

    return {"ok": True, "kind": "memory_store", "stored": True, "memory": obj}


def recent_memories(limit: int = 20) -> List[Dict[str, Any]]:
    load_memory_cache()
    return list(reversed(_memory_cache[-max(1, min(limit, 100)):]))


def clear_memories() -> int:
    global _memory_cache, _memory_mtime, _memory_version
    ensure_memory_dir()
    with _memory_lock:
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            return 0
        removed = len(_memory_cache)
        _memory_cache = []
        _memory_version += 1
        try:
            _memory_mtime = os.path.getmtime(MEMORY_FILE)
        except OSError:
            _memory_mtime = 0.0
        return removed


def forget_memory(memory_id: str) -> bool:
    global _memory_cache, _memory_mtime, _memory_version
    target = str(memory_id or "").strip()
    if not target:
        return False
    load_memory_cache()
    with _memory_lock:
        original = list(_memory_cache)
        new_rows = [row for row in original if str(row.get("id", "")) != target]
        if len(new_rows) == len(original):
            return False
        ensure_memory_dir()
        tmp = MEMORY_FILE + f".tmp-{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for row in new_rows:
                    f.write(compact_json(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, MEMORY_FILE)
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
        _memory_cache = new_rows
        _memory_version += 1
        try:
            _memory_mtime = os.path.getmtime(MEMORY_FILE)
        except OSError:
            _memory_mtime = 0.0
        return True


# ============================================================
# FILE TOOLS
# ============================================================

def resolve_path(path: str) -> str:
    """Resolve a workspace path and reject symlink/path traversal outside WORKSPACE."""
    raw = os.path.expanduser(str(path or "."))
    if not os.path.isabs(raw):
        raw = os.path.join(WORKSPACE, raw)
    target = os.path.realpath(raw)
    root = WORKSPACE_REAL
    try:
        if os.path.commonpath([root, target]) != root:
            raise ValueError("path outside workspace")
    except ValueError:
        raise ValueError("path outside workspace")
    return target


def read_file(path: str) -> Dict[str, Any]:
    target = resolve_path(path)
    with open(target, "rb") as f:
        data = f.read(MAX_FILE_READ)
    return {
        "ok": True,
        "kind": "read_file",
        "path": target,
        "bytes": len(data),
        "content": data.decode("utf-8", errors="replace"),
    }


def write_file(path: str, content: str) -> Dict[str, Any]:
    target = resolve_path(path)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    data = str(content).encode("utf-8")
    fd, temp = tempfile.mkstemp(prefix=".braincells-tmp-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, target)
    finally:
        try:
            if os.path.exists(temp):
                os.unlink(temp)
        except OSError:
            pass
    return {
        "ok": True,
        "kind": "write_file",
        "path": target,
        "bytes": len(data),
    }


def append_file(path: str, content: str) -> Dict[str, Any]:
    target = resolve_path(path)
    if os.path.isdir(target):
        raise IsADirectoryError(target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    data = str(content).encode("utf-8")
    with open(target, "ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return {
        "ok": True,
        "kind": "append_file",
        "path": target,
        "bytes": len(data),
    }


def replace_text(path: str, old: str, new: str) -> Dict[str, Any]:
    target = resolve_path(path)
    if not old:
        return {"ok": False, "kind": "replace_text", "path": target, "error": "The old text must be non-empty."}
    if os.path.isdir(target):
        return {"ok": False, "kind": "replace_text", "path": target, "error": "Cannot replace text in a directory."}
    with open(target, "r", encoding="utf-8", errors="replace") as f:
        original = f.read()
    count = original.count(old)
    if count == 0:
        return {"ok": False, "kind": "replace_text", "path": target, "error": "Exact old text was not found."}
    updated = original.replace(old, new)
    write_file(target, updated)
    return {
        "ok": True,
        "kind": "replace_text",
        "path": target,
        "replacements": count,
        "bytes": len(updated.encode("utf-8")),
    }


def delete_file(path: str) -> Dict[str, Any]:
    target = resolve_path(path)
    if target == WORKSPACE_REAL:
        return {"ok": False, "kind": "delete_file", "path": target, "error": "Refusing to delete the workspace root."}
    if os.path.isdir(target):
        return {"ok": False, "kind": "delete_file", "path": target, "error": "Refusing to delete directories; delete specific files instead."}
    os.remove(target)
    return {"ok": True, "kind": "delete_file", "path": target}


def list_files(path: str) -> Dict[str, Any]:
    target = resolve_path(path)
    root = WORKSPACE_REAL
    entries = []
    for name in sorted(os.listdir(target)):
        try:
            full = os.path.join(target, name)
            resolved = os.path.realpath(full)
            if os.path.commonpath([root, resolved]) != root:
                continue
            entries.append({
                "name": name,
                "type": "dir" if os.path.isdir(full) else "file",
                "size": os.path.getsize(full) if os.path.isfile(full) else None,
            })
        except OSError:
            entries.append({"name": name, "type": "unknown", "size": None})
    return {"ok": True, "kind": "list_files", "path": target, "entries": entries[:2000], "count": len(entries)}


def search_files(query: str, path: str) -> Dict[str, Any]:
    root = resolve_path(path or WORKSPACE)
    q = str(query or "").strip().lower()
    if not q:
        return {"ok": False, "kind": "search_files", "error": "query is required"}

    hits: List[Dict[str, Any]] = []

    # Fast path: ripgrep is dramatically faster on real repositories; keep the pure-Python fallback below.
    # `shutil` was missing in v3, which made this call raise NameError before it ever reached
    # either backend — search_files was completely dead. Fixed here.
    rg = shutil.which("rg")
    if rg:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [rg, "-n", "-i", "-F", "--no-heading", "--color", "never", "--json",
                 "--hidden", "--glob", "!.git/**", "--glob", "!node_modules/**",
                 "--glob", "!.venv/**", "--glob", "!venv/**", "--", q, root],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                timeout=min(COMMAND_TIMEOUT, 20), check=False, env=_safe_child_env(),
            )
            # --json gives exact path/line/text fields instead of guessing at colon positions
            # in filenames that legitimately contain colons.
            for raw in proc.stdout.splitlines():
                if len(hits) >= 120:
                    break
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(event, dict) or event.get("type") != "match":
                    continue
                data = event.get("data")
                if not isinstance(data, dict):
                    continue
                path_field = data.get("path")
                path_text = path_field.get("text", "") if isinstance(path_field, dict) else str(path_field or "")
                lines_field = data.get("lines")
                lines = lines_field.get("text", "") if isinstance(lines_field, dict) else str(lines_field or "")
                try:
                    line_no = int(data.get("line_number") or 0)
                except (TypeError, ValueError):
                    line_no = 0
                try:
                    rel = os.path.relpath(path_text, root)
                except Exception:
                    rel = path_text
                hits.append({"file": rel, "line": line_no, "text": clip(lines.rstrip("\n"), 800)})
            return {
                "ok": proc.returncode in (0, 1),
                "kind": "search_files",
                "query": q,
                "results": hits,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "backend": "rg",
                "error": clip(proc.stderr.strip(), 1000) if proc.returncode not in (0, 1) else None,
            }
        except (OSError, subprocess.TimeoutExpired):
            pass
    started = time.monotonic()
    ignored = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    # Compiled once per call instead of lowercasing every single line of every file.
    needle = re.compile(re.escape(q), re.I)

    def walk(directory: str):
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for entry in entries:
            if entry.name in ignored:
                continue
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    yield from walk(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    yield entry.path
            except OSError:
                continue

    for fp in walk(root):
        try:
            if os.path.getsize(fp) > MAX_FILE_READ:
                continue
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    if needle.search(line):
                        hits.append({
                            "file": os.path.relpath(fp, root),
                            "line": line_no,
                            "text": clip(line.rstrip(), 800),
                        })
                        if len(hits) >= 120:
                            return {
                                "ok": True, "kind": "search_files", "query": q,
                                "results": hits, "backend": "python",
                                "duration_ms": round((time.monotonic() - started) * 1000),
                            }
        except (OSError, UnicodeError):
            continue

    return {
        "ok": True, "kind": "search_files", "query": q, "results": hits,
        "backend": "python",
        "duration_ms": round((time.monotonic() - started) * 1000),
    }


# ============================================================
# SHELL
# ============================================================

_SECRET_ENV_NAME = re.compile(
    r"(^|_)(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH(?:ORIZATION)?|PASSWORD|PASSWD|SECRET|PRIVATE[_-]?KEY|CREDENTIAL|OAUTH|AUTHKEY|SIGNING[_-]?KEY)(_|$)",
    re.I,
)

def _safe_child_env() -> Dict[str, str]:
    env = dict(os.environ)
    explicit = {
        "TOOL_WIELDER_KEY", "TAILSCALE_AUTHKEY", "TS_OAUTH_SECRET",
        "TS_OAUTH_CLIENT_SECRET", "GITHUB_TOKEN", "GH_TOKEN",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    }
    for key in list(env):
        if key in explicit or _SECRET_ENV_NAME.search(key):
            env.pop(key, None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env

def run_command(command: str, cwd: Optional[str]) -> Dict[str, Any]:
    workdir = resolve_path(cwd or WORKSPACE)
    started = time.monotonic()
    process: Optional[subprocess.Popen[str]] = None

    try:
        process = subprocess.Popen(
            str(command),
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
            executable=os.environ.get("SHELL", "/bin/sh"),
            env=_safe_child_env(),
        )
        try:
            stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            # A child stuck in uninterruptible (D) state can outlive even SIGKILL, so the
            # reap must itself be bounded — otherwise the CLI hangs forever waiting on a
            # process that will never report. Two shots, then give up and return what we have.
            stdout, stderr = "", ""
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

        return {
            "ok": process.returncode == 0 and not timed_out,
            "kind": "execute",
            "command": str(command),
            "cwd": workdir,
            "pid": process.pid,
            "exit_code": process.returncode,
            "stdout": clip(stdout, 20000),
            "stderr": clip(stderr, 12000),
            "timed_out": timed_out,
            "duration_ms": round((time.monotonic() - started) * 1000),
        }
    except Exception as exc:
        return {
            "ok": False,
            "kind": "execute",
            "command": str(command),
            "cwd": workdir,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_ms": round((time.monotonic() - started) * 1000),
        }


# ============================================================
# WEB
# ============================================================

def clean_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unwrap_search_url(href: str) -> str:
    href = html.unescape(str(href or "")).strip()
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        for key in ("uddg", "url", "u"):
            if query.get(key):
                candidate = unquote(query[key][0])
                if candidate.startswith(("http://", "https://")):
                    return candidate
    except Exception:
        pass
    return href


class _DDGParser(HTMLParser):
    """Tolerant parser for DuckDuckGo result links/snippets; avoids brittle attribute-order regexes."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[Tuple[str, str]] = []
        self.snippets: List[str] = []
        self._capture_link = False
        self._capture_snippet = False
        self._link_href = ""
        self._link_text: List[str] = []
        self._snippet_text: List[str] = []
        self._depth = 0

    @staticmethod
    def _classes(attrs: List[Tuple[str, Optional[str]]]) -> str:
        return " ".join((v or "") for k, v in attrs if k == "class").lower()

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        classes = self._classes(attrs)
        if tag == "a" and "result__a" in classes:
            self._capture_link = True
            self._link_href = next((v or "" for k, v in attrs if k == "href"), "")
            self._link_text = []
            self._depth = 1
        elif self._capture_link:
            self._depth += 1
        if (tag == "a" or tag == "div") and "result__snippet" in classes:
            self._capture_snippet = True
            self._snippet_text = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture_link:
            if tag == "a" and self._depth == 1:
                title = re.sub(r"\s+", " ", "".join(self._link_text)).strip()
                if title and self._link_href:
                    self.links.append((unwrap_search_url(self._link_href), title))
                self._capture_link = False
                self._depth = 0
            elif self._depth > 0:
                self._depth -= 1
        if self._capture_snippet and tag in {"a", "div"}:
            snippet = re.sub(r"\s+", " ", "".join(self._snippet_text)).strip()
            if snippet:
                self.snippets.append(snippet)
            self._capture_snippet = False

    def handle_data(self, data: str) -> None:
        if self._capture_link:
            self._link_text.append(data)
        if self._capture_snippet:
            self._snippet_text.append(data)


def _parse_ddg(page: str, max_results: int) -> List[Dict[str, Any]]:
    parser = _DDGParser()
    try:
        parser.feed(page)
    except Exception:
        return []
    results: List[Dict[str, Any]] = []
    for idx, (url, title) in enumerate(parser.links[:max_results]):
        if not url.startswith(("http://", "https://")):
            continue
        results.append({
            "rank": len(results) + 1,
            "title": clip(title, 300),
            "url": clip(url, 1200),
            "snippet": clip(parser.snippets[idx] if idx < len(parser.snippets) else "", 900),
        })
    return results


def _parse_bing(page: str, max_results: int) -> List[Dict[str, Any]]:
    # Bing's result structure is comparatively stable; keep this as a fallback only.
    pattern = re.compile(
        r'<li[^>]+class=["\'][^"\']*b_algo[^"\']*["\'][^>]*>.*?'
        r'<h2[^>]*>\s*<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>.*?'
        r'(?:<p[^>]*>(.*?)</p>)?.*?</li>',
        re.I | re.S,
    )
    results: List[Dict[str, Any]] = []
    for match in pattern.finditer(page):
        url = unwrap_search_url(match.group(1))
        title = clean_html(match.group(2))
        snippet = clean_html(match.group(3) or "")
        if not title or not url.startswith(("http://", "https://")):
            continue
        results.append({
            "rank": len(results) + 1,
            "title": clip(title, 300),
            "url": clip(url, 1200),
            "snippet": clip(snippet, 900),
        })
        if len(results) >= max_results:
            break
    return results


def _search_http(url: str) -> Tuple[int, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
        },
    )
    with urlopen(request, timeout=WEB_TIMEOUT) as response:
        raw = response.read(MAX_WEB_BYTES)
        return int(response.status), raw.decode("utf-8", errors="replace")


def _run_search_backend(backend: str, encoded: str, max_results: int) -> Tuple[str, List[Dict[str, Any]], str]:
    """Fetch+parse one backend. Returns (name, results, error)."""
    try:
        if backend == "ddg":
            status, page = _search_http(f"https://html.duckduckgo.com/html/?{encoded}")
            if status >= 400:
                raise RuntimeError(f"DuckDuckGo HTTP {status}")
            results = _parse_ddg(page, max_results)
            if not results:
                return "duckduckgo", [], "DuckDuckGo returned no parseable results"
            return "duckduckgo", results, ""
        if backend == "bing":
            status, page = _search_http(f"https://www.bing.com/search?{encoded}")
            if status >= 400:
                raise RuntimeError(f"Bing HTTP {status}")
            results = _parse_bing(page, max_results)
            if not results:
                return "bing", [], "Bing returned no parseable results"
            return "bing", results, ""
        return backend, [], f"unknown backend: {backend}"
    except Exception as exc:
        return backend, [], f"{backend}: {type(exc).__name__}: {exc}"


def web_search(query: str, max_results: int = 6) -> Dict[str, Any]:
    query = str(query).strip()[:500]
    max_results = max(1, min(int(max_results), 10))
    if not query:
        return {"ok": False, "kind": "web_search", "error": "query is required"}

    key = f"{query.lower()}::{max_results}"
    now = time.monotonic()
    with _web_lock:
        cached = _web_cache.get(key)
        if cached and now - cached[0] < WEB_CACHE_TTL and cached[1].get("results"):
            result = dict(cached[1])
            result["cached"] = True
            return result

    started = time.monotonic()
    errors: List[str] = []
    backends = [x.strip().lower() for x in SEARCH_BACKENDS.split(",") if x.strip()]
    if not backends:
        backends = ["ddg", "bing"]

    results: List[Dict[str, Any]] = []
    used_backend = ""
    encoded = urlencode({"q": query})

    # Racing the backends instead of running them back-to-back turns the worst case
    # from (ddg_timeout + bing_timeout) into roughly a single timeout.
    if SEARCH_PARALLEL and len(backends) > 1:
        with ThreadPoolExecutor(max_workers=min(len(backends), 4)) as pool:
            outcomes = list(pool.map(lambda b: _run_search_backend(b, encoded, max_results), backends))
        # Honour configured priority: report the first backend that produced results.
        for name, found, err in outcomes:
            if found and not results:
                results, used_backend = found, name
            if err:
                errors.append(err)
    else:
        for backend in backends:
            name, found, err = _run_search_backend(backend, encoded, max_results)
            if found:
                results, used_backend = found, name
                break
            if err:
                errors.append(err)

    duration_ms = round((time.monotonic() - started) * 1000)
    if not results:
        return {
            "ok": False,
            "kind": "web_search",
            "query": query,
            "results": [],
            "count": 0,
            "cached": False,
            "duration_ms": duration_ms,
            "error": "; ".join(errors[-4:]) or "No search results returned.",
        }

    result = {
        "ok": True,
        "kind": "web_search",
        "query": query,
        "results": results,
        "count": len(results),
        "backend": used_backend,
        "cached": False,
        "duration_ms": duration_ms,
    }
    with _web_lock:
        _web_cache[key] = (now, result)
        if len(_web_cache) > 128:
            oldest_key = min(_web_cache, key=lambda k: _web_cache[k][0])
            _web_cache.pop(oldest_key, None)
    return result


_dns_lock = threading.Lock()
_dns_cache: Dict[str, Tuple[float, List[Any]]] = {}


def _resolve_host(host: str, port: int) -> List[Any]:
    """getaddrinfo is a blocking syscall that can cost hundreds of milliseconds; every
    redirect hop used to pay for it again. Cache the answer briefly instead."""
    key = f"{host}:{port}"
    now = time.monotonic()
    with _dns_lock:
        hit = _dns_cache.get(key)
        if hit is not None and now - hit[0] < DNS_CACHE_TTL:
            return hit[1]
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    with _dns_lock:
        if len(_dns_cache) > 512:
            _dns_cache.clear()
        _dns_cache[key] = (now, infos)
    return infos


def _is_non_public_ip(addr: str) -> bool:
    """True for loopback/private/link-local/multicast/reserved/exotic addresses."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved
    )


def _connected_peer_ip(response: Any) -> Optional[str]:
    """Best-effort read of the IP a live HTTP response is actually connected to.

    The check in `_public_http_url` validates the address the hostname *resolved* to; a
    rebinding DNS server can hand urllib a different, private address on the second lookup.
    Comparing the post-connect peer closes that TOCTOU window.
    """
    probes = (
        lambda r: r.fp.raw._sock,
        lambda r: r.fp.raw._sock._sock,
        lambda r: r.fp._sock,
    )
    for probe in probes:
        try:
            sock = probe(response)
            return sock.getpeername()[0]
        except Exception:
            continue
    return None


def _public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported.")
    if not parsed.hostname:
        raise ValueError("URL hostname is missing.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded usernames/passwords are refused.")
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = _resolve_host(host, port)
    except OSError as exc:
        raise ValueError(f"DNS lookup failed: {exc}") from exc
    for info in infos:
        addr = info[4][0]
        if _is_non_public_ip(addr):
            raise ValueError("Refusing to fetch private or local network addresses.")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

_SAFE_WEB_OPENER = build_opener(_NoRedirect())

def _safe_open_url(url: str, timeout: float, max_hops: int = 4):
    current = url
    for _ in range(max_hops + 1):
        # Validation happens here, once per hop — fetch_url must NOT pre-validate the
        # same URL or the first hostname is resolved twice.
        _public_http_url(current)
        request = Request(
            current,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "identity",
            },
        )
        try:
            response = _SAFE_WEB_OPENER.open(request, timeout=timeout)
            peer = _connected_peer_ip(response)
            if peer and _is_non_public_ip(peer):
                # Resolved-public but connected-private == DNS rebinding. Refuse the body.
                response.close()
                raise ValueError(
                    "Refusing to read from a private or local network address."
                )
            return response, current
        except HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            location = exc.headers.get("Location")
            exc.close()
            if not location:
                raise ValueError(f"Redirect {exc.code} had no Location header")
            current = urljoin(current, location)
    raise ValueError("Too many redirects while fetching URL.")

def fetch_url(url: str) -> Dict[str, Any]:
    target = str(url or "").strip()
    if not target:
        return {"ok": False, "kind": "fetch_url", "error": "url is required"}

    started = time.monotonic()
    try:
        response, checked_url = _safe_open_url(target, FETCH_TIMEOUT)
        with response:
            raw = response.read(MAX_WEB_BYTES)
            content_type = response.headers.get("content-type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            final_url = response.geturl() or checked_url
    except Exception as exc:
        return {
            "ok": False,
            "kind": "fetch_url",
            "url": target,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_ms": round((time.monotonic() - started) * 1000),
        }

    try:
        # get_content_charset() can report a charset Python has no codec for; an unknown
        # codec raises LookupError from decode and used to escape this function's error path.
        text = raw.decode(charset, errors="replace")
    except (LookupError, UnicodeError):
        text = raw.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in text[:2000].lower():
        text = clean_html(text)
    return {
        "ok": True,
        "kind": "fetch_url",
        "url": final_url,
        "content_type": content_type,
        "text": clip(text, 18000),
        "bytes": len(raw),
        "duration_ms": round((time.monotonic() - started) * 1000),
    }


# ============================================================
# WORKER DISPATCH
# ============================================================

def dispatch_worker(req: Dict[str, Any], qwen: QwenOllama) -> Dict[str, Any]:
    kind = str(req.get("kind", "")).strip()
    instruction = str(req.get("instruction", "")).strip()
    context = str(req.get("context", "")).strip()

    try:
        if kind == "execute":
            return run_command(str(req.get("command") or instruction), req.get("cwd") or WORKSPACE)

        if kind == "read_file":
            return read_file(str(req.get("path") or instruction))

        if kind == "write_file":
            path = str(req.get("path") or "").strip()
            if not path:
                return {"ok": False, "kind": kind, "error": "path is required"}
            if "content" not in req:
                return {"ok": False, "kind": kind, "error": "content is required"}
            return write_file(path, str(req.get("content")))

        if kind == "append_file":
            path = str(req.get("path") or "").strip()
            if not path:
                return {"ok": False, "kind": kind, "error": "path is required"}
            return append_file(path, str(req.get("content", "")))

        if kind == "replace_text":
            path = str(req.get("path") or "").strip()
            if not path:
                return {"ok": False, "kind": kind, "error": "path is required"}
            return replace_text(path, str(req.get("old", "")), str(req.get("new", "")))

        if kind == "delete_file":
            return delete_file(str(req.get("path") or instruction))

        if kind == "list_files":
            return list_files(str(req.get("path") or instruction or WORKSPACE))

        if kind == "search_files":
            return search_files(
                str(req.get("query") or instruction),
                str(req.get("path") or WORKSPACE),
            )

        if kind == "web_search":
            return web_search(
                str(req.get("query") or instruction),
                int(req.get("max_results") or 6),
            )

        if kind == "fetch_url":
            return fetch_url(str(req.get("url") or instruction))

        if kind == "review":
            system = str(req.get("system") or "")
            prompt = str(req.get("prompt") or "")
            max_predict = int(req.get("max_predict") or 512)
            data = qwen.json_chat(system, prompt, max_predict)
            return {"ok": True, "kind": kind, "data": data if isinstance(data, dict) else {}}

        if kind == "memory_retrieve":
            return memory_retrieve(instruction, qwen)

        if kind == "memory_store":
            return memory_store(instruction, context, qwen)

        return {"ok": False, "kind": kind, "error": f"Unknown tool kind: {kind}"}
    except Exception as exc:
        return {
            "ok": False,
            "kind": kind,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# WORKER HTTP SERVER
# ============================================================

class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "BraincellsToolWielder/3"
    protocol_version = "HTTP/1.1"
    # Idle/partial clients must not pin a worker thread forever (each is a daemon thread;
    # 8MB bodies over a slow tailnet still send a chunk well inside 30s).
    timeout = 30

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        data = compact_json(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The caller hung up (cancelled probe, tailnet blip). There is nobody left to
            # receive this payload, and re-raising just prints a traceback from socketserver.
            pass

    def _authorized(self) -> bool:
        return bool(TOOL_WIELDER_KEY) and hmac.compare_digest(
            self.headers.get("Authorization", ""),
            f"Bearer {TOOL_WIELDER_KEY}",
        )

    def do_GET(self) -> None:
        if not self._authorized():
            self._json(401, {"ok": False, "error": "unauthorized"})
            return
        route = urlparse(self.path).path
        if route == "/health":
            load_memory_cache()
            self._json(200, {
                "ok": True,
                "service": "tool-wielder",
                "model": QWEN_MODEL,
                "memory_count": len(_memory_cache),
                "workspace": WORKSPACE,
            })
            return
        if route == "/config":
            self._json(200, {
                "ok": True,
                "service": "tool-wielder",
                "model": QWEN_MODEL,
                "listen": f"{os.getenv('WORKER_LISTEN_HOST', '0.0.0.0')}:{TOOL_PORT}",
                "workspace": WORKSPACE,
                "memory_file": MEMORY_FILE,
                "limits": {
                    "command_timeout": COMMAND_TIMEOUT,
                    "max_file_read": MAX_FILE_READ,
                    "max_web_bytes": MAX_WEB_BYTES,
                },
            })
            return
        if route == "/memories":
            rows = recent_memories(20)
            self._json(200, {"ok": True, "count": len(rows), "memories": rows})
            return
        if route == "/warm":
            # Compatibility: older clients sometimes probe /warm with GET. Warm is idempotent.
            try:
                qwen = getattr(self.server, "qwen", None)
                if qwen is None:
                    raise RuntimeError("Qwen backend is not initialized")
                started = time.monotonic()
                qwen.warm()
                self._json(200, {"ok": True, "service": "tool-wielder", "model": QWEN_MODEL, "warmed": True, "duration_ms": round((time.monotonic() - started) * 1000)})
            except Exception as exc:
                self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        if route == "/help":
            self._json(200, {
                "ok": True,
                "endpoints": {
                    "GET": ["/health", "/config", "/help", "/memories"],
                    "POST": ["/warm", "/reset", "/clear-memories", "/v1/act"],
                },
                "actions": sorted([
                    "execute", "read_file", "write_file", "append_file",
                    "replace_text", "delete_file", "list_files", "search_files",
                    "web_search", "fetch_url", "memory_retrieve", "memory_store",
                    "review"
                ]),
            })
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"ok": False, "error": "unauthorized"})
            return
        route = urlparse(self.path).path
        if route == "/reset":
            with _web_lock:
                _web_cache.clear()
            self._json(200, {"ok": True, "reset": True, "message": "transient worker caches reset; persistent memory retained"})
            return

        if route == "/clear-memories":
            removed = clear_memories()
            self._json(200, {"ok": True, "removed": removed})
            return

        if route == "/warm":
            try:
                qwen = getattr(self.server, "qwen", None)
                if qwen is None:
                    raise RuntimeError("Qwen backend is not initialized")
                started = time.monotonic()
                qwen.warm()
                self._json(200, {
                    "ok": True,
                    "service": "tool-wielder",
                    "model": QWEN_MODEL,
                    "warmed": True,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                })
            except Exception as exc:
                self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return

        if route != "/v1/act":
            self._json(404, {"ok": False, "error": "not found"})
            return

        try:
            raw_len = int(self.headers.get("Content-Length", "0"))
            if raw_len <= 0 or raw_len > MAX_REQUEST_BODY:
                self._json(413, {"ok": False, "error": "invalid request size"})
                return
            raw = self.rfile.read(raw_len)
            req = json.loads(raw)
            if not isinstance(req, dict):
                self._json(400, {"ok": False, "error": "request must be a JSON object"})
                return

            started = time.monotonic()
            qwen = getattr(self.server, "qwen", None)
            if qwen is None:
                raise RuntimeError("Qwen backend is not initialized")
            result = dispatch_worker(req, qwen)
            result["task_id"] = req.get("task_id")
            result["duration_ms"] = round((time.monotonic() - started) * 1000)
            self._json(200, result)
        except Exception as exc:
            self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class BraincellsHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ============================================================
# TOOL CLIENT
# ============================================================

class ToolClient:
    """HTTP client for the Tool Wielder backed by a small connection pool.

    v3 kept ONE persistent connection guarded by a lock, so even if the caller had
    parallelised tool calls they would have queued up single-file on that lock. The pool
    is what turns "concurrent" into concurrent — the worker is a ThreadingHTTPServer and
    handles them side by side.
    """

    def __init__(self, host: str, port: int, key: str, timeout: float = 120):
        self.hostport = f"{host}:{port}"
        self.timeout = timeout
        self.key = key
        self._pool: "queue.LifoQueue[PersistentHTTP]" = queue.LifoQueue()
        for _ in range(max(1, TOOL_CONN_POOL)):
            self._pool.put(PersistentHTTP(self.hostport, timeout))
        # Backstop for anything not going through the pool (health/warm/memories).
        self.http = PersistentHTTP(self.hostport, timeout)

    @contextmanager
    def _lease(self):
        conn = self._pool.get(timeout=self.timeout)
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def close(self) -> None:
        seen = []
        while True:
            try:
                seen.append(self._pool.get_nowait())
            except queue.Empty:
                break
        seen.append(self.http)
        for conn in seen:
            try:
                conn.close()
            except Exception:
                pass

    def health(self) -> Dict[str, Any]:
        with self._lease() as conn:
            return json.loads(conn.request(
                "GET",
                "/health",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.key}",
                    "Connection": "keep-alive",
                },
            ))

    def warm(self) -> Dict[str, Any]:
        with self._lease() as conn:
            return json.loads(conn.request(
                "POST",
                "/warm",
                body=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.key}",
                    "Connection": "keep-alive",
                },
            ))

    def memories(self) -> Dict[str, Any]:
        with self._lease() as conn:
            return json.loads(conn.request(
                "GET", "/memories",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.key}",
                    "Connection": "keep-alive",
                },
            ))

    def clear_memories(self) -> Dict[str, Any]:
        with self._lease() as conn:
            return json.loads(conn.request(
                "POST", "/clear-memories", body=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.key}",
                    "Connection": "keep-alive",
                },
            ))

    def call(self, task_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        body = compact_json({"task_id": task_id, **args}).encode("utf-8")
        with self._lease() as conn:
            return json.loads(conn.request(
                "POST",
                "/v1/act",
                body,
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.key}",
                    "Connection": "keep-alive",
                },
            ))


class RemoteQwen:
    """Qwen seen from the client side: reached through the Tool Wielder, not localhost.

    The client box has no Ollama — Gemma is remote and Qwen lives on the worker — so the
    review step borrows the worker's model over the connection we already authenticate.
    It keeps json_chat() so qwen_review() is identical whichever side runs it.
    """

    def __init__(self, worker: ToolClient):
        self._worker = worker

    def json_chat(self, system: str, user: str, max_predict: int) -> Dict[str, Any]:
        out = self._worker.call("review", {
            "kind": "review",
            "system": system,
            "prompt": user,
            "max_predict": max_predict,
        })
        if not isinstance(out, dict) or out.get("ok") is False:
            return {}
        data = out.get("data")
        return data if isinstance(data, dict) else {}

    def close(self) -> None:
        pass


# ============================================================
# RESULT COMPACTION / TOOL LOOP
# ============================================================

# The fields that actually eat Gemma's window. When the extractor replaces a result these
# are what it gets to drop; everything else in the envelope is kept, so a reduced result
# still says what it was, where it came from and whether it worked.
BULKY_RESULT_KEYS = frozenset({
    "stdout", "stderr", "content", "text", "results", "entries", "memories", "matches",
})


def _result_envelope(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only information the Main Brain can actually use, minimizing context growth."""
    kind = result.get("kind")
    if kind == "write_file":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "path": result.get("path"),
            "bytes": result.get("bytes"),
            "error": result.get("error"),
        }
    elif kind == "execute":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "exit_code": result.get("exit_code"),
            "stdout": clip(result.get("stdout"), 6000),
            "stderr": clip(result.get("stderr"), 4000),
            "timed_out": result.get("timed_out"),
            "duration_ms": result.get("duration_ms"),
            "error": result.get("error"),
        }
    elif kind == "read_file":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "path": result.get("path"),
            "bytes": result.get("bytes"),
            "content": clip(result.get("content"), 12000),
            "error": result.get("error"),
        }
    elif kind == "search_files":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "query": result.get("query"),
            # `or []`: get(key, []) only defaults when the key is ABSENT; a tool that emits
            # {"results": null} would make None[:40] raise and take the whole turn down.
            "results": (result.get("results") or [])[:40],
            "error": result.get("error"),
        }
    elif kind == "web_search":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "query": result.get("query"),
            "count": result.get("count"),
            "backend": result.get("backend"),
            "cached": result.get("cached"),
            "results": (result.get("results") or [])[:6],
            "error": result.get("error"),
        }
    elif kind == "fetch_url":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "url": result.get("url"),
            "content_type": result.get("content_type"),
            "text": clip(result.get("text"), 12000),
            "error": result.get("error"),
        }
    elif kind == "memory_retrieve":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "terms": result.get("terms"),
            "memories": (result.get("memories") or [])[:6],
            "error": result.get("error"),
        }
    elif kind == "memory_store":
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "stored": result.get("stored"),
            "memory": result.get("memory"),
            "error": result.get("error"),
        }
    else:
        reduced = {
            "ok": result.get("ok"),
            "kind": kind,
            "error": result.get("error"),
            "path": result.get("path"),
            "entries": (result.get("entries") or [])[:200],
            "stored": result.get("stored"),
        }
    return reduced


def compact_tool_result(result: Dict[str, Any],
                        reduced: Optional[Dict[str, Any]] = None) -> str:
    """Assemble the tool message: Qwen's verdict first, then the evidence for it.

    `reduced` lets a caller hand in an envelope the extractor has already shrunk. The
    review is attached HERE rather than inside the per-kind reduction, so shrinking a
    result can never silently drop the one sentence Gemma most needs to read.
    """
    envelope = dict(reduced) if isinstance(reduced, dict) else _result_envelope(result)
    # Everything Gemma cannot re-derive — what Qwen concluded, and which brain ran it —
    # goes in front, because the clip only ever eats the tail.
    prefix: Dict[str, Any] = {}
    review = result.get("review")
    if isinstance(review, dict):
        prefix["review"] = {
            "status": review.get("status"),
            "diagnosis": clip(review.get("diagnosis"), 600),
            "fix": clip(review.get("fix"), 600),
        }
    if result.get("executor") == "local":
        prefix["executor"] = "local"
    if prefix:
        envelope = {**prefix, **envelope}
    return clip(compact_json(envelope), MAX_TOOL_RESULT)


def extract_tool_result(
    result: Dict[str, Any],
    label: str,
    needle: Optional[NeedleRouter],
) -> Optional[Dict[str, Any]]:
    """Shrink an oversized result and return the envelope to feed back into compaction.

    Only the bulky fields are surrendered: `ok`, `kind`, `path` and friends stay, so the
    message Gemma reads still says the call worked and where the file came from.
    """
    reduced_text = needle_extract(label, compact_tool_result(result), needle)
    if not reduced_text:
        return None
    try:
        parsed = json.loads(reduced_text)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    envelope = {k: v for k, v in _result_envelope(result).items()
                if k not in BULKY_RESULT_KEYS}
    envelope.update(parsed)
    return envelope


def tool_call_fields(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    fn = call.get("function") or {}
    name = str(fn.get("name") or "")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return name, args if isinstance(args, dict) else {}


TOOL_HINTS = re.compile(
    r"(?i)\b(?:build|create|write|edit|modify|change|fix|debug|run|execute|test|verify|read|open|save|delete|remove|move|copy|list|search|find|fetch|browse|web|url|curl|wget|git|install|compile|deploy|script|program|memory|remember|forget)\b"
)


def likely_tool_task(text: str) -> bool:
    text = str(text).strip()
    if not text:
        return False
    if len(text) > 120:
        return True
    return bool(TOOL_HINTS.search(text))


def tool_call_key(name: str, args: Dict[str, Any]) -> str:
    return compact_json({"name": name, "args": args})


def _group_turns(history: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for msg in history:
        if msg.get("role") == "system":
            continue
        if msg.get("role") == "user" and current:
            groups.append(current)
            current = []
        current.append(msg)
    if current:
        groups.append(current)
    return groups


CONTINUATION_HINTS = re.compile(r"(?i)\b(?:continue|keep\s+going|carry\s+on|finish(?:\s+it)?|resume|proceed|go\s+on)\b")

def looks_like_continuation(text: str) -> bool:
    return bool(CONTINUATION_HINTS.search(str(text or "")))

def trim_history_turns(history: List[Dict[str, Any]], keep_turns: int) -> List[Dict[str, Any]]:
    groups = _group_turns(history)
    chosen = groups[-max(1, keep_turns):]
    flattened = [m for group in chosen for m in group]
    head = history[0] if history and history[0].get("role") == "system" else {
        "role": "system", "content": SYSTEM_PROMPT}
    return [head, *flattened]


# ---- Gemma-driven memory compaction --------------------------------------------------
# Mechanical trimming drops old turns and the model forgets them forever. Instead, when the
# session grows past MAX_HISTORY we let the CEO (Gemma) read the turns that are about to be
# discarded and write the ONLY thing about them worth keeping — a short priority-memory
# note that is re-injected on every later call. It runs only when history is genuinely too
# large, never per-round, and on any hiccup it falls back to the exact trim it replaced.

COMPACT_SYSTEM = (
    "You are the memory librarian for a tool-using agent. You read finished conversation "
    "turns and write the single note that should survive them."
)


def _gemma_compact_turns(brain: MainOllama, turns: List[List[Dict[str, Any]]]) -> Optional[str]:
    """Ask the CEO to condense finished turns into one priority-memory note.

    Never raises meaningful failures: every path returns None, and the caller falls back
    to the mechanical trim. The input is itself a digest (messages clipped hard), so the
    compaction call is a genuinely lightweight task, not a re-read of the session.
    """
    lines: List[str] = []
    for idx, group in enumerate(turns, 1):
        parts: List[str] = []
        for msg in group:
            role = msg.get("role")
            text = " ".join(str(msg.get("content") or "").split())
            if not text:
                continue
            if role == "tool":
                parts.append(f"[tool] {text[:200]}")
            elif role == "assistant":
                if msg.get("tool_calls"):
                    parts.append(f"[assistant tool] {text[:140]}")
                else:
                    parts.append(f"[assistant] {text[:160]}")
            else:
                parts.append(f"[user] {text[:160]}")
        if parts:
            lines.append(f"Turn {idx}: " + " | ".join(parts))
    digest = clip("\n".join(lines), COMPACT_CONTEXT_MAX)
    if not digest.strip():
        return None

    user_txt = (
        "The session outgrew its window, so these finished turns are about to be archived. "
        "Write the ONE note the agent will still need about them. Keep concrete facts: file "
        "paths, commands chosen, open goals, constraints, decisions and their reasons, and "
        "mistakes to avoid. Drop finished narration and dead ends. Concrete is better than "
        "clever. Return ONLY a JSON object of the form "
        '{{"memory":"<your note, at most {maxc} characters>"}}'.format(maxc=COMPACT_MEMORY_MAX)
    )
    try:
        response = brain.chat(
            [
                {"role": "system", "content": COMPACT_SYSTEM},
                {"role": "user", "content": user_txt + "\n\n" + digest},
            ],
            use_tools=False,
            num_predict=min(COMPACT_MEMORY_MAX + 200, MAIN_MAX_OUTPUT),
            num_ctx=MAIN_CTX,
            temperature=0.0,
        )
    except Exception:
        return None
    content = str(((response.get("message") or {}).get("content") or "")).strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(content[start:end + 1])
    except Exception:
        return None
    memory = str(parsed.get("memory") or "").strip()
    if not memory:
        return None
    return clip(memory, COMPACT_MEMORY_MAX)


def _write_memory_drawer(groups_to_drop: List[List[Dict[str, Any]]],
                         note: Optional[str]) -> None:
    """Append archived turns to a JSONL drawer before compaction drops them from the prompt.

    Compaction is a *prompt* optimization, not a data policy: the model no longer carries the
    raw turns, but the user can still recover them from disk. Best-effort and fail-open — a
    drawer write must never block or fail a turn.
    """
    if not groups_to_drop:
        return
    try:
        os.makedirs(os.path.dirname(MEMORY_DRAWER), exist_ok=True)
        record = {
            "ts": time.time(),
            "note": note,
            "turns": [
                [
                    {"role": m.get("role"),
                     "content": clip(str(m.get("content") or ""), COMPACT_CONTEXT_MAX)}
                    for m in group
                ]
                for group in groups_to_drop
            ],
        }
        with open(MEMORY_DRAWER, "a", encoding="utf-8") as f:
            f.write(compact_json(record) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def maybe_compact_history(history: List[Dict[str, Any]],
                          brain: Optional[MainOllama]) -> List[Dict[str, Any]]:
    """When history exceeds MAX_HISTORY turns, compact what is about to be dropped.

    The trigger is purely mechanical (history is objectively too large); the CEO then
    decides *what* matters — the cheap gate fires the smart librarian. Any failure keeps
    the untouched mechanical trim, so this can never make history worse.
    """
    groups = _group_turns(history)
    if len(groups) <= MAX_HISTORY:
        return history
    droppable = groups[:len(groups) - MAX_HISTORY]
    keep = groups[len(groups) - MAX_HISTORY:]

    note: Optional[str] = None
    if COMPACT_MEMORY and brain is not None:
        note = _gemma_compact_turns(brain, droppable)

    # Archive the raw turns first, whether or not the summary succeeded: the drawer, not the
    # in-prompt memory, is the lossless copy.
    _write_memory_drawer(droppable, note)

    if note:
        with _STATE_LOCK:
            _session_memory.append(note)
            if COMPACT_MEMORY_KEEP > 0:
                del _session_memory[:-COMPACT_MEMORY_KEEP]
        lead = history[0] if history and history[0].get("role") == "system" else {
            "role": "system", "content": SYSTEM_PROMPT}
        flattened = [m for group in keep for m in group]
        return [lead, *flattened]

    # Fallback is byte-for-byte the v4 behaviour.
    return trim_history_turns(history, MAX_HISTORY)

def tool_calls_similar(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ka = str(a.get("kind") or "").strip().lower()
    kb = str(b.get("kind") or "").strip().lower()
    if ka != kb:
        return False
    if ka == "web_search":
        ta = set(re.findall(r"[a-z0-9]{3,}", str(a.get("query") or "").lower()))
        tb = set(re.findall(r"[a-z0-9]{3,}", str(b.get("query") or "").lower()))
        if not ta or not tb:
            return False
        return len(ta & tb) / max(1, len(ta | tb)) >= 0.65
    if ka == "fetch_url":
        return str(a.get("url") or "").strip().rstrip("/") == str(b.get("url") or "").strip().rstrip("/")
    if ka in {"read_file", "list_files", "delete_file", "write_file", "append_file", "replace_text", "execute"}:
        return compact_json(a) == compact_json(b)
    return False

def _message_chars(msg: Dict[str, Any]) -> int:
    return len(str(msg.get("content") or ""))


def _fit_group_to_budget(group: List[Dict[str, Any]], budget_chars: int, shrink_to: int = 1200) -> List[Dict[str, Any]]:
    """Keep the newest messages of one turn; compress older ones instead of losing them.

    One rule, applied newest-first: keep whole if it fits, otherwise keep a truncated copy
    if that fits, otherwise stop. Truncating (rather than dropping) older tool results means
    the model still knows the operation ran and how it ended — it just loses the bulk.
    Returned messages are copies, so the caller's history is never mutated.
    """
    if not group:
        return group
    lead = group[0]
    used = _message_chars(lead)
    tail: List[Dict[str, Any]] = []
    for msg in reversed(group[1:]):
        size = _message_chars(msg)
        if used + size <= budget_chars:
            tail.append(msg)
            used += size
            continue
        cap = min(shrink_to, budget_chars - used)
        if cap < 200:
            break
        copy = dict(msg)
        copy["content"] = clip(str(msg.get("content") or ""), cap)
        added = _message_chars(copy)
        if used + added > budget_chars:
            break
        tail.append(copy)
        used += added
    tail.reverse()
    kept = [lead, *tail]
    # The cut happens at the oldest end and can strand a `tool` result whose assistant
    # tool_calls message fell outside the budget. Ollama pairs them positionally, so drop
    # the stranded results rather than hand over an orphaned tool message.
    while len(kept) > 1 and kept[1].get("role") == "tool":
        kept.pop(1)
    return kept


def _fit_groups_to_budget(groups: List[List[Dict[str, Any]]], budget_chars: int) -> List[List[Dict[str, Any]]]:
    """Drop whole oldest turns first so the current task always survives.

    v3 sent whatever `TOOL_HISTORY_TURNS` happened to select, which could blow straight
    past num_ctx once tool results stacked up — Ollama then truncates or rejects the call.
    Fitting to a character budget keeps the request inside the context window instead.
    """
    if not groups:
        return groups
    if budget_chars <= 0:
        return groups[-1:]
    chosen: List[List[Dict[str, Any]]] = []
    used = 0
    for group in reversed(groups):
        size = sum(_message_chars(m) for m in group)
        if chosen and used + size > budget_chars:
            break
        if not chosen and size > budget_chars:
            chosen.append(_fit_group_to_budget(group, budget_chars))
            break
        chosen.append(group)
        used += size
    chosen.reverse()
    return chosen or groups[-1:]


def _memory_block() -> str:
    """The priority-memory distilled by Gemma from compacted turns, injected into every call."""
    if not _session_memory:
        return ""
    keep = max(1, COMPACT_MEMORY_KEEP)
    notes = "\n".join(
        f"[Priority memory {i + 1}] {note}"
        for i, note in enumerate(_session_memory[-keep:])
    )
    return notes


def _system_prompt_for(use_tools: bool, base: str) -> str:
    """The system prompt for a call, with compacted priority memory appended when present.

    Memory is a summary Gemma itself wrote from turns that have ALREADY been consumed, so
    feeding it back costs a handful of tokens and can never hide in-flight work. It changes
    only when a compaction runs — not per round — so the system prefix (system + tool schema)
    stays byte-stable across the rounds of a task and Ollama's prompt-prefix cache keeps
    hitting. Only the per-round grounding hint lives at the message tail.
    """
    block = _memory_block()
    if block:
        return base + "\n\n----------\nKEEP IN MIND (committed memory from earlier turns):\n" + block
    return base


def _context_tail(extra: Optional[str]) -> Optional[str]:
    """Per-call dynamic context that rides at the very end of the message list.

    The grounding hint changes every synthesis call, so it must sit after the cached prefix
    (system + memory + history). It is kept as one optional message so the conversation
    shape and token budget stay predictable.
    """
    return extra or None


def _prepare_history_for_call(
    history: List[Dict[str, Any]],
    use_tools: bool,
    *,
    current_only: bool = False,
    num_ctx: int = MAIN_CTX,
    extra: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a small valid Ollama conversation that fits the requested context window.

    The tail context is counted against the budget too: a grounding hint that pushes the
    request past num_ctx would be silently truncated by Ollama, so it has to be reserved for
    up front rather than added on top of a budget that assumed it was not there.
    """
    groups = _group_turns(history)
    tail = _context_tail(extra)
    tail_chars = len(tail or "")
    if not groups:
        built = [{
            "role": "system",
            "content": _system_prompt_for(use_tools, SYSTEM_PROMPT if use_tools else CHAT_SYSTEM_PROMPT),
        }]
        return built + ([{"role": "user", "content": tail}] if tail else [])

    if use_tools:
        system_prompt = _system_prompt_for(use_tools, SYSTEM_PROMPT)
        limit = 1 if (current_only and groups) else TOOL_HISTORY_TURNS
        selected = groups[-limit:]
        budget = int(num_ctx * CHARS_PER_TOKEN) - (len(system_prompt) + tail_chars + 512)
        # A multi-round task is one logical turn: _group_turns already keeps every tool
        # message of the current task together, so the budget only ever sheds OLDER turns.
        selected = _fit_groups_to_budget(selected, budget)
        flattened = [m for group in selected for m in group]
        built = [{"role": "system", "content": system_prompt}, *flattened]
        return built + ([{"role": "user", "content": tail}] if tail else [])

    # Tool internals are not useful during ordinary chat and can confuse a no-tools call.
    system_prompt = _system_prompt_for(use_tools, CHAT_SYSTEM_PROMPT)
    budget = int(min(num_ctx, CHAT_CTX) * CHARS_PER_TOKEN) - (len(system_prompt) + tail_chars + 512)
    selected = _fit_groups_to_budget(groups[-CHAT_HISTORY_TURNS:], budget)
    flattened: List[Dict[str, Any]] = []
    for msg in (m for group in selected for m in group):
        role = msg.get("role")
        if role == "tool":
            continue
        if role == "assistant" and msg.get("tool_calls"):
            continue
        flattened.append({"role": role, "content": str(msg.get("content") or "")})
    built = [{"role": "system", "content": system_prompt}, *flattened]
    return built + ([{"role": "user", "content": tail}] if tail else [])


def _semantic_tool_key(name: str, args: Dict[str, Any]) -> str:
    """Normalize repeat-prone calls so model wording changes do not cause duplicate work."""
    kind = str(args.get("kind") or "").strip().lower()
    if kind in {"web_search", "fetch_url"}:
        value = args.get("query") if kind == "web_search" else args.get("url")
        return compact_json({"name": name, "kind": kind, "value": re.sub(r"\s+", " ", str(value or "").strip().lower())})
    if kind in {"read_file", "list_files", "delete_file"}:
        return compact_json({"name": name, "kind": kind, "path": os.path.normpath(str(args.get("path") or args.get("instruction") or "").strip())})
    if kind == "execute":
        return tool_call_key(name, args)
    if kind == "write_file":
        content = str(args.get("content") or "")
        return compact_json({
            "name": name,
            "kind": kind,
            "path": os.path.normpath(str(args.get("path") or "").strip()),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    if kind == "append_file":
        content = str(args.get("content") or "")
        return compact_json({
            "name": name,
            "kind": kind,
            "path": os.path.normpath(str(args.get("path") or "").strip()),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    if kind == "replace_text":
        return tool_call_key(name, args)
    return tool_call_key(name, args)


def _gemma_call_with_retry(
    brain: MainOllama,
    history: List[Dict[str, Any]],
    *,
    use_tools: bool,
    num_predict: int,
    num_ctx: int,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    last: Optional[BaseException] = None
    for attempt in range(2):
        try:
            return brain.chat(
                history,
                use_tools=use_tools,
                num_predict=num_predict,
                num_ctx=num_ctx,
                temperature=temperature,
            )
        except HTTPFailure as exc:
            last = exc
            if exc.status not in {502, 503, 504} or attempt:
                raise
            time.sleep(0.25)
        except (ConnectionError, BrokenPipeError, OSError, socket.timeout) as exc:
            last = exc
            if attempt:
                raise
            time.sleep(0.15)
    raise RuntimeError(str(last))


# ============================================================
# EXECUTION PIPELINE
#   Gemma authors the script -> Needle grounds + runs it -> Qwen reads the output
#   -> the analysis goes back to Gemma, which decides end / continue / add tasks.
# ============================================================

class _NoQwen:
    """Stand-in so dispatch_worker() still runs when no Qwen brain is configured."""

    def json_chat(self, system: str, user: str, max_predict: int) -> Dict[str, Any]:
        return {}


def _qwen_or_stub(qwen: Any) -> Any:
    return qwen if qwen is not None else _NoQwen()


def _result_text(result: Dict[str, Any]) -> str:
    """The part of a tool result Qwen should actually read."""
    parts: List[str] = []
    for key in ("stdout", "stderr", "content", "error"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}:\n{value.strip()}")
    for key in ("results", "memories", "entries", "files", "matches"):
        value = result.get(key)
        if isinstance(value, list) and value:
            parts.append(f"{key}: {compact_json(value[:25])}")
    return clip("\n".join(parts) or compact_json(result), 6000)


QWEN_REVIEW_SYSTEM = (
    "You are the execution reviewer in a three-brain agent. The CEO (Gemma) wrote a tool "
    "command, it was executed, and you are reading what came back.\n"
    "Return JSON only, no prose, exactly:\n"
    '{"status":"ok"|"failed"|"partial"|"no_change",'
    '"diagnosis":"one literal sentence describing what the output actually shows",'
    '"fix":"one concrete next instruction for the CEO, empty when status is ok"}\n'
    "Judge the OUTPUT, not the intent. A non-zero exit code, an exception, a traceback, or "
    "an explicit error field means failed. A successful run that changed nothing is "
    "no_change. Never repeat the output back. Never invent an error that is not in it."
)


def qwen_review(
    args: Dict[str, Any],
    result: Dict[str, Any],
    qwen: Any,
    max_predict: int = 260,
) -> Optional[Dict[str, Any]]:
    """Qwen reads EVERY tool output — success or failure — and reports status back to Gemma.

    Returns {"status", "diagnosis", "fix"} or None. A missing or dead Qwen simply means
    Gemma sees the raw result, which is exactly how v4 behaved. Transport failures are
    allowed to propagate so the caller's circuit breaker can see them; an answer that came
    back but was empty/unparsable still returns None.
    """
    if not QWEN_REVIEW or qwen is None:
        return None
    payload = {
        "kind": str(args.get("kind") or result.get("kind") or "unknown"),
        "instruction": clip(
            str(args.get("instruction") or args.get("path") or args.get("query")
                or args.get("url") or ""),
            600,
        ),
        "ok": bool(result.get("ok")),
        "exit_code": result.get("exit_code"),
        "duration_ms": result.get("duration_ms"),
        "output": _result_text(result),
    }
    data = qwen.json_chat(QWEN_REVIEW_SYSTEM, compact_json(payload), max_predict)
    if not isinstance(data, dict) or not data:
        return None
    status = str(data.get("status") or "").strip().lower()
    if status not in {"ok", "failed", "partial", "no_change"}:
        # A model that ignored the enum still earns a read: derive it from ok/failure.
        if result.get("ok"):
            status = "ok"
        else:
            status = "failed"
    review = {
        "status": status,
        "diagnosis": clip(str(data.get("diagnosis") or ""), 600),
        "fix": clip(str(data.get("fix") or ""), 600),
    }
    if not review["diagnosis"] and not review["fix"]:
        return None
    return review


QWEN_REVIEW_BATCH_SYSTEM = (
    "You are the execution reviewer in a three-brain agent. The CEO (Gemma) issued a batch "
    "of tool commands, they were executed, and you are reading every result in order.\n"
    "Return JSON only, no prose, exactly:\n"
    '{"reviews":[{"index":N,'
    '"status":"ok"|"failed"|"partial"|"no_change",'
    '"diagnosis":"one literal sentence describing what that output actually shows",'
    '"fix":"one concrete next instruction for the CEO, empty when status is ok"}]}\n'
    "Give exactly one entry per index. Judge each OUTPUT, not the intent. A non-zero exit "
    "code, an exception, a traceback, or an explicit error field means failed. A successful "
    "run that changed nothing is no_change. Never repeat output back. Never invent an error "
    "that is not in it."
)


def qwen_review_batch(planned: List[Dict[str, Any]], qwen: Any, max_predict: int = 900) -> bool:
    """One Qwen call for the WHOLE round instead of one per result.

    Every result still gets the same {status, diagnosis, fix} verdict — it is the same
    reviewer reading the same outputs — but a multi-call round now pays one model call and
    one network round trip instead of N, and Qwen sees every output together so fixes for
    later calls can lean on earlier ones. Returns True only when every batch item got a
    review; otherwise the caller keeps the old per-item path.
    """
    batch: List[Dict[str, Any]] = []
    for index, item in enumerate(planned):
        result = item.get("result")
        if not isinstance(result, dict) or not result or "review" in result:
            continue
        args = item.get("args") or {}
        batch.append({
            "index": index,
            "kind": str(args.get("kind") or result.get("kind") or "unknown"),
            "instruction": clip(
                str(args.get("instruction") or args.get("path") or args.get("query")
                    or args.get("url") or ""),
                600,
            ),
            "ok": bool(result.get("ok")),
            "exit_code": result.get("exit_code"),
            "duration_ms": result.get("duration_ms"),
            "output": clip(_result_text(result), 1400),
        })
    if not batch:
        return False
    data = qwen.json_chat(QWEN_REVIEW_BATCH_SYSTEM, compact_json({"results": batch}), max_predict)
    if not isinstance(data, dict):
        return False
    reviews = data.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        return False
    by_index: Dict[int, Dict[str, Any]] = {}
    for entry in reviews:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        # Accept 1.0 as well as 1: JSON numbers arrive as floats often enough that an
        # integer-only check silently drops perfectly good verdicts.
        if isinstance(index, bool) or not isinstance(index, (int, float)):
            continue
        index = int(index)
        if not (0 <= index < len(planned)):
            continue
        by_index[index] = entry
    attached = 0
    for item, report in ((planned[b["index"]], b) for b in batch):
        result = item.get("result")
        if not isinstance(result, dict) or "review" in result:
            continue
        verdict = by_index.get(report["index"])
        if not isinstance(verdict, dict):
            continue
        status = str(verdict.get("status") or "").strip().lower()
        if status not in {"ok", "failed", "partial", "no_change"}:
            status = "ok" if result.get("ok") else "failed"
        review = {
            "status": status,
            "diagnosis": clip(str(verdict.get("diagnosis") or ""), 600),
            "fix": clip(str(verdict.get("fix") or ""), 600),
        }
        if review["diagnosis"] or review["fix"]:
            result["review"] = review
            attached += 1
    return attached == len(batch)


def needle_scope(user_text: str, needle: Optional[NeedleRouter]) -> Optional[bool]:
    """True when the request is a tool task, False when it is not, None when unsure.

    A confident False stops Gemma from wasting tool rounds on plain conversation. It can
    never *block* an answer: the worst case is Gemma replying directly, which is correct.
    """
    if needle is None or not NEEDLE_REFUSER or not needle.has("refuser"):
        return None
    result = needle.ask(
        "refuser",
        system=(
            "Decide whether answering this request requires calling a tool "
            "(run a command, touch a file, search, fetch, memory). Pure conversation, "
            "explanations and opinions do not. Submit exactly one `submit` call."
        ),
        user=clip(user_text, 2000),
        schema=NEEDLE_SCOPE_SCHEMA,
        max_tokens=120,
    )
    if not result:
        return None
    if result.get("confidence") is None or result["confidence"] < needle.threshold("refuser"):
        return None
    for call in result.get("calls", []):
        if call["name"] == "submit":
            return bool(call["arguments"].get("tool_task"))
    return None


def needle_route(user_text: str, needle: Optional[NeedleRouter]) -> Optional[str]:
    """Which tool kind handles this? Only ever consulted when Gemma proposed none.

    It cannot override a decision, because it is never reached when Gemma hands back a
    real tool call. What it can do is rescue a round that would otherwise end the turn
    with `tool_mode` still on and nothing executed — and `kind="none"` lets it decline.
    """
    if needle is None or not NEEDLE_ROUTER or not needle.has("router"):
        return None
    result = needle.ask(
        "router",
        system=(
            "Given one user request, pick the single Tool Wielder kind that should "
            "handle it. Submit exactly one `submit` call. Choose `none` when no tool "
            "applies or a tool would be inappropriate for the request."
        ),
        user=clip(user_text, 2000),
        schema=NEEDLE_ROUTE_SCHEMA,
        max_tokens=120,
    )
    if not result:
        return None
    if result.get("confidence") is None or result["confidence"] < needle.threshold("router"):
        return None
    for call in result.get("calls", []):
        if call["name"] == "submit":
            kind = str(call["arguments"].get("kind") or "")
            return kind if kind in TOOL_KINDS else None
    return None


def needle_ground(
    calls: List[Dict[str, Any]],
    needle: Optional[NeedleRouter],
) -> Optional[List[Dict[str, Any]]]:
    """Needle turns each of Gemma's decisions into complete, concrete arguments.

    Gemma owns the work; the executioner may only FILL IN what the CEO left blank.
    The merge below is strictly additive and kind-preserving, and every structural
    check that fails returns None so Gemma's original calls execute untouched.
    """
    if (
        needle is None
        or not NEEDLE_PLANNER
        or not needle.has("planner")
        or not calls
    ):
        return None

    decisions = []
    for index, call in enumerate(calls):
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, dict):
            continue
        decisions.append({
            "index": index,
            "kind": args.get("kind"),
            "instruction": args.get("instruction"),
            "supplied": {k: v for k, v in args.items() if k not in {"kind", "instruction"}},
        })
    if not decisions:
        return None

    result = needle.ask(
        "planner",
        system=(
            "You are the execution layer. The Main Brain has ALREADY decided what must "
            "happen; you only express each decision as complete tool arguments.\n"
            "Return exactly one `tool_wielder` call per decision, in the same order and "
            "with the same index. Keep `kind` identical to the decision. Never restate or "
            "change what was decided — only supply the concrete values the CEO left blank "
            "(path, query, content, command, cwd, url). Anything already in `supplied` is "
            "authoritative: reproduce it verbatim."
        ),
        user=compact_json({"decisions": decisions}),
        tools=MAIN_TOOLS,
        max_tokens=NEEDLE_MAX_TOKENS,
    )
    if not result:
        return None
    if result.get("confidence") is None or result["confidence"] < needle.threshold("planner"):
        return None

    grounded = result.get("calls") or []
    # One call per decision, or the executioner misread the script — decline the offer.
    if len(grounded) != len(decisions):
        return None
    if any(call["name"] != "tool_wielder" for call in grounded):
        return None

    merged: List[Dict[str, Any]] = []
    for decision, call in zip(decisions, grounded):
        # Copy both levels: a shallow copy would leave original["function"] pointing at
        # Gemma's own dict, and writing the filled arguments would edit the CEO's script.
        original = dict(calls[decision["index"]])
        original_fn = dict(original.get("function") or {})
        original_args = original_fn.get("arguments")
        if not isinstance(original_args, dict):
            return None
        proposed = call.get("arguments")
        if not isinstance(proposed, dict):
            return None
        if proposed.get("kind") != original_args.get("kind"):
            return None                      # the executioner may not re-target the axe
        filled = dict(original_args)
        for key, value in proposed.items():
            if key in filled and filled[key] not in (None, "", [], {}):
                continue                     # the CEO's word wins
            filled[key] = value
        original_fn["arguments"] = filled
        original["function"] = original_fn
        merged.append(original)
    return merged


def needle_verify(
    instruction: str,
    results: List[Dict[str, Any]],
    needle: Optional[NeedleRouter],
) -> Optional[Dict[str, Any]]:
    """Did the executed work actually fulfil Gemma's instruction? Returns {satisfied, gap}."""
    if needle is None or not NEEDLE_VERIFIER or not needle.has("verifier"):
        return None
    outcome = [
        {"ok": r.get("ok"), "kind": r.get("kind"), "output": clip(_result_text(r), 1500)}
        for r in results
    ]
    result = needle.ask(
        "verifier",
        system=(
            "You are checking whether an executed tool instruction was actually fulfilled. "
            "Compare the instruction against the outputs. Submit exactly one `submit` call. "
            "Report `satisfied` true only when the outputs clearly do what was asked; "
            "otherwise state the specific gap."
        ),
        user=compact_json({"instruction": clip(instruction, 800), "outputs": outcome}),
        schema=NEEDLE_VERIFY_SCHEMA,
        max_tokens=160,
    )
    if not result:
        return None
    if result.get("confidence") is None or result["confidence"] < needle.threshold("verifier"):
        return None
    for call in result.get("calls", []):
        if call["name"] == "submit":
            args = call["arguments"]
            return {
                "satisfied": bool(args.get("satisfied")),
                "gap": clip(str(args.get("gap") or ""), 400),
            }
    return None


def needle_draft_fix(
    instruction: str,
    kind: str,
    gap: str,
    needle: Optional[NeedleRouter],
) -> Optional[Dict[str, Any]]:
    """Needle turns Qwen's one-line fix into one COMPLETE, executable retry call.

    The result is a *proposal*, attached to the tool message Gemma reads next: Gemma keeps
    final say over whether to run it, edit it, or ignore it. A successful draft means the
    next CEO round is a short "approve / adjust" instead of a from-scratch re-plan.
    """
    if needle is None or not DRAFT_FIX or not NEEDLE_PLANNER or not needle.has("planner"):
        return None
    result = needle.ask(
        "planner",
        system=(
            "The Main Brain's last attempt failed. Draft exactly ONE concrete "
            "`tool_wielder` call that applies the fix. Keep `kind` identical to the failed "
            "call. Supply complete arguments (path, command, query, content, cwd, url) — "
            "do not leave anything blank."
        ),
        user=compact_json({
            "failed_instruction": clip(instruction, 400),
            "kind": str(kind or "unknown"),
            "gap": clip(gap, 300),
        }),
        tools=MAIN_TOOLS,
        max_tokens=NEEDLE_MAX_TOKENS,
    )
    if not result:
        return None
    if result.get("confidence") is None or result["confidence"] < needle.threshold("planner"):
        return None
    for call in result.get("calls", []):
        if call["name"] != "tool_wielder":
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            continue
        if str(args.get("kind") or "").strip().lower() != str(kind or "").strip().lower():
            continue
        return dict(args)
    return None


def needle_extract(
    instruction: str,
    content: str,
    needle: Optional[NeedleRouter],
) -> Optional[str]:
    """Shrink an oversized tool result into structured fields before it reaches Gemma."""
    if needle is None or not NEEDLE_EXTRACTOR or not needle.has("extractor"):
        return None
    if len(content) < NEEDLE_EXTRACT_MIN:
        return None
    result = needle.ask(
        "extractor",
        system=(
            "Reduce a large tool output to only what answers the instruction. Keep every "
            "concrete value — paths, line numbers, names, numbers, statuses, error text — "
            "and drop everything else. Submit exactly one `submit` call."
        ),
        user=compact_json({
            "instruction": clip(instruction, 600),
            "output": clip(content, 24000),
        }),
        schema=NEEDLE_EXTRACT_SCHEMA,
        max_tokens=NEEDLE_EXTRACT_MAX,
    )
    if not result:
        return None
    if result.get("confidence") is None or result["confidence"] < needle.threshold("extractor"):
        return None
    for call in result.get("calls", []):
        if call["name"] != "submit":
            continue
        args = call["arguments"]
        summary = str(args.get("summary") or "").strip()
        facts = [str(f).strip() for f in (args.get("facts") or []) if str(f).strip()]
        if not summary and not facts:
            return None
        reduced = compact_json({"summary": summary, "facts": facts[:20]})
        # Never trade a smaller context for a bigger one.
        if len(reduced) >= len(content):
            return None
        return reduced
    return None


def _worker_exec(
    worker: Optional[ToolClient],
    item: Dict[str, Any],
    qwen: Any = None,
    local: bool = False,
) -> Dict[str, Any]:
    """Run one planned call, then let Qwen read what came back.

    `local` runs the script in this process (no network hop); otherwise it goes to the
    remote Tool Wielder. Both paths produce the same result shape and both are reviewed.
    """
    task_id = f"t{time.time_ns()}"
    started = time.monotonic()
    args = item.get("args") or {}
    kind = str(item.get("kind") or args.get("kind") or "").strip()
    # The Tool Wielder owns the memory store, so those two never move: running them here
    # would quietly split one brain's memory across two machines. Everything else is fair
    # game for the local executor.
    if kind in ("memory_retrieve", "memory_store"):
        local = False
    result: Dict[str, Any]

    if local:
        try:
            result = dict(dispatch_worker(args, _qwen_or_stub(qwen)))
        except Exception as exc:
            result = {
                "ok": False,
                "kind": item.get("kind") or "unknown",
                "task_id": task_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result.setdefault("kind", item.get("kind") or result.get("kind"))
        result["executor"] = "local"
    else:
        try:
            if worker is None:
                raise RuntimeError("no Tool Wielder configured")
            result = dict(worker.call(task_id, args))
        except Exception as exc:
            result = {
                "ok": False,
                "kind": item.get("kind") or "unknown",
                "task_id": task_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result.setdefault("kind", item.get("kind") or result.get("kind"))
        result["executor"] = "remote"

    if SHOW_TIMINGS:
        println(f"{DIM}↳ {result['executor']} "
                f"{round((time.monotonic() - started) * 1000)} ms{RESET}")
    return result


def _final_grounding(tool_messages: List[str], user_text: str) -> str:
    """A compact tail message that hands Gemma the tool evidence a tool-free round hides.

    The final synthesis round runs with use_tools=False, which strips every tool message
    from call history. Without this, Gemma anthropomorphically "remembers" the tool work
    while reading none of it. A few clipped lines ground the answer in the results.
    """
    head = f'Tool work for "{clip(user_text, 200)}" produced:'
    return head + "\n" + "\n\n".join(tool_messages[-6:])


def review_planned(planned: List[Dict[str, Any]], qwen: Any) -> None:
    """Qwen's pass over every result of the round — success and failure alike.

    Layers, newest first:
      1. Session review cache — an output+instruction Qwen already approved "ok" is
         reused; a changed result (different digest) always goes through the reviewer.
      2. One batched call for the whole round (same reviewer, one model call).
      3. Per-item fallback if the batch came back malformed.

    Transport failures trip the circuit breaker (fail-open for the session) instead of
    being re-tried and re-timing-out on every round.
    """
    if not planned or qwen is None or not QWEN_REVIEW:
        return
    if not _breaker_ready("review:qwen"):
        return
    try:
        for item in planned:
            result = item.get("result")
            if not isinstance(result, dict) or "review" in result:
                continue
            hit = _review_cache_hit(item.get("args") or {}, result)
            if hit is not None:
                result["review"] = hit

        if qwen_review_batch(planned, qwen):
            for item in planned:
                result = item.get("result")
                if isinstance(result, dict) and result.get("review"):
                    _review_cache_store(item.get("args") or {}, result, result["review"])
            _breaker_ok("review:qwen")
            return

        for item in planned:
            result = item.get("result")
            if not isinstance(result, dict) or "review" in result:
                continue
            review = qwen_review(item.get("args") or {}, result, qwen)
            if review:
                result["review"] = review
                _review_cache_store(item.get("args") or {}, result, review)
        _breaker_ok("review:qwen")
    except HTTPFailure as exc:
        # The worker answered, so it is reachable: an HTTP error here is the review ACTION
        # failing, not a dead endpoint. If it says the action is unknown (an older Tool
        # Wielder with no review branch), stop asking — one clear line beats a bogus
        # "degraded" breaker warning every round. A generic 5xx (overload) still trips the
        # breaker so later rounds do not repay it.
        if _review_capability_error(getattr(exc, "body", "")):
            _disable_review(
                f"the Tool Wielder at {TOOL_HOST}:{TOOL_PORT} rejected the review action "
                f"({clip(str(getattr(exc, 'body', '')), 140)})"
            )
        else:
            _breaker_fail("review:qwen", f"{type(exc).__name__}: {exc}")
    except (ConnectionError, BrokenPipeError, OSError, socket.timeout) as exc:
        # Unreachable reviewer: leave results unreviewed (v4 behaviour) and trip the
        # breaker so the next rounds do not re-pay the same timeout.
        _breaker_fail("review:qwen", f"{type(exc).__name__}: {exc}")
    except ValueError:
        # Unparsable reviewer JSON is transient, not a dead endpoint: keep the breaker
        # closed so the next round gets a fresh shot.
        pass


def _plan_tool_calls(
    calls: List[Dict[str, Any]],
    seen_calls: Dict[str, Dict[str, Any]],
    prior_call_args: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate and de-duplicate every requested call BEFORE anything executes.

    Splitting planning from execution is what makes batching safe: by the time we decide
    what may run concurrently, we already know which calls are pure reads and which are
    cache hits that need no worker round-trip at all.
    """
    planned: List[Dict[str, Any]] = []
    for call in calls:
        name, args = tool_call_fields(call)
        item: Dict[str, Any] = {
            "name": name, "args": args, "kind": "unknown", "key": None,
            "result": None, "needs_worker": False, "duplicate": False,
        }
        if name != "tool_wielder":
            item["result"] = {
                "ok": False, "kind": "unknown",
                "error": f"Unknown tool function: {name or '<empty>'}",
            }
            planned.append(item)
            continue

        kind = str(args.get("kind") or "").strip().lower()
        instruction = str(args.get("instruction") or "").strip()
        if not kind:
            item["result"] = {"ok": False, "kind": "unknown", "error": "Missing tool kind."}
            planned.append(item)
            continue
        item["kind"] = kind
        if not instruction and kind not in {"list_files"}:
            item["result"] = {"ok": False, "kind": kind, "error": "Tool instruction is empty."}
            planned.append(item)
            continue

        key = _semantic_tool_key(name, args)
        item["key"] = key
        println(f"{DIM}↳ {kind}: {clip(instruction or args.get('path') or '', 220)}{RESET}")

        # Exact cache first; then a semantic near-duplicate check for repetitive web/search calls.
        prior_key = None
        for old_args in prior_call_args[-64:]:
            if tool_calls_similar(old_args, args):
                prior_key = _semantic_tool_key("tool_wielder", old_args)
                break

        hit_key = key if seen_calls.get(key, {}).get("ok") else None
        if hit_key is None and prior_key and seen_calls.get(prior_key, {}).get("ok"):
            hit_key = prior_key

        if hit_key is not None:
            result = dict(seen_calls[hit_key])
            result["cached_duplicate"] = True
            result["duplicate_of"] = hit_key
            item["result"] = result
            item["duplicate"] = True
            label = "near-duplicate" if hit_key != key else "duplicate"
            println(f"{DIM}↳ {label} avoided; prior successful result reused{RESET}")
        else:
            item["needs_worker"] = True
        prior_call_args.append(dict(args))
        planned.append(item)
    return planned


def _execute_planned(
    planned: List[Dict[str, Any]],
    worker: Optional[ToolClient],
    qwen: Any = None,
    local: bool = False,
) -> None:
    """Run planned calls in stated order, concurrently batching consecutive read-only ones.

    A side-effecting call acts as a barrier: everything before it finishes first, so the
    executor still observes exactly the order the model asked for. Read-only calls have no
    observable effect, so a run of them can safely overlap — a 3-call round then costs the
    slowest call rather than the sum of all three.
    """
    total = len(planned)
    i = 0
    while i < total:
        item = planned[i]
        if item["result"] is not None or not item["needs_worker"]:
            i += 1
            continue
        if item["kind"] not in READ_ONLY_KINDS:
            item["result"] = _worker_exec(worker, item, qwen, local)
            i += 1
            continue
        j = i + 1
        while j < total and planned[j]["needs_worker"] and planned[j]["kind"] in READ_ONLY_KINDS:
            j += 1
        batch = planned[i:j]
        if len(batch) > 1 and TOOL_PARALLEL:
            with ThreadPoolExecutor(max_workers=min(len(batch), 6)) as pool:
                futures = [
                    pool.submit(_worker_exec, worker, it, qwen, local) for it in batch
                ]
                for it, future in zip(batch, futures):
                    it["result"] = future.result()
        else:
            for it in batch:
                it["result"] = _worker_exec(worker, it, qwen, local)
        i = j


def run_turn(
    user_text: str,
    history: List[Dict[str, Any]],
    brain: MainOllama,
    worker: Optional[ToolClient],
    on_delta: Optional[Callable[[str], None]] = None,
    needle: Optional[NeedleRouter] = None,
    qwen: Any = None,
) -> str:
    """Three-brain loop: Gemma plans and answers, Needle executes, Qwen reports back.

    It stops on completion or genuine no-progress, not on task length. Every Needle and
    Qwen step is optional and fails open, so an unconfigured or unreachable local brain
    degrades to exactly the two-brain v4 behaviour.
    """
    history.append({"role": "user", "content": user_text})
    seen_calls: Dict[str, Dict[str, Any]] = {}
    prior_call_args: List[Dict[str, Any]] = []
    tool_mode = likely_tool_task(user_text) or looks_like_continuation(user_text)
    # Needle's scope gate can only ever *save* Gemma tool rounds. A confident "not a tool
    # task" means answer it directly; anything else leaves Gemma's own judgement alone.
    if tool_mode and needle_scope(user_text, needle) is False:
        tool_mode = False
    force_no_tools_next = False
    rescued = False
    stalled_rounds = 0
    final_synthesis = False
    round_no = 1
    emitted = 0
    last_round_tools: List[str] = []

    while True:
        exhausted = (
            (MAX_TOOL_ROUNDS > 0 and round_no > MAX_TOOL_ROUNDS)
            or round_no > MAX_ROUNDS_HARD
        )
        if exhausted and not final_synthesis:
            # Graceful degradation: take one tool-free pass at an answer rather than
            # reporting a round-count error the user never asked about.
            final_synthesis = True
            force_no_tools_next = True
        elif exhausted:
            break

        gen_started = time.monotonic()
        use_tools = tool_mode and not force_no_tools_next

        # num_predict is a CAP, not a target: Ollama stops at EOS, so a generous cap is
        # free on short replies and is the difference between a valid write_file payload
        # and a truncated one on long ones. v3's 512-token router cap was a real failure mode.
        if use_tools:
            predict, ctx, temperature = MAIN_MAX_OUTPUT, MAIN_CTX, MAIN_TEMPERATURE
        else:
            predict = CHAT_MAX_OUTPUT
            # num_ctx is an Ollama *load* option: any change rebuilds the runner, re-prefills
            # and discards the KV cache. So the loaded context is pinned to MAIN_CTX for every
            # call; the prompt is still trimmed to a CHAT_CTX-sized budget inside
            # _prepare_history_for_call, we just never ask Ollama to reload to get there.
            ctx = MAIN_CTX
            temperature = SYNTH_TEMPERATURE

        streaming = bool(on_delta is not None and STREAM and not use_tools)

        try:
            call_history = _prepare_history_for_call(
                history,
                use_tools=use_tools,
                current_only=(round_no > 1),
                num_ctx=ctx,
                extra=_final_grounding(last_round_tools, user_text) if (
                    final_synthesis and last_round_tools) else None,
            )
            # On the first continuation call, preserve the previous incomplete turn explicitly.
            if round_no == 1 and looks_like_continuation(user_text):
                call_history = _prepare_history_for_call(
                    history, use_tools=True, current_only=False, num_ctx=ctx
                )
            if streaming:
                def _delta(piece: str) -> None:
                    nonlocal emitted
                    emitted += 1
                    on_delta(piece)  # type: ignore[misc]

                response = brain.chat_stream(
                    call_history,
                    use_tools=use_tools,
                    num_predict=predict,
                    num_ctx=ctx,
                    temperature=temperature,
                    on_delta=_delta,
                )
            else:
                response = _gemma_call_with_retry(
                    brain,
                    call_history,
                    use_tools=use_tools,
                    num_predict=predict,
                    num_ctx=ctx,
                    temperature=temperature,
                )
        except Exception as exc:
            message = f"Main Brain connection error: {type(exc).__name__}: {exc}"
            if emitted:
                # Tokens are already on screen; ending the line beats swallowing them.
                println()
                eprintln(message)
                return ""
            if round_no == 1 and history and history[-1].get("role") == "user":
                # Nothing generated yet: drop the dangling user turn so the next prompt is
                # not a user→user pair with no assistant reply between them.
                history.pop()
            return message

        gen_ms = round((time.monotonic() - gen_started) * 1000)
        if SHOW_TIMINGS:
            println(f"{DIM}[Gemma r{round_no} {gen_ms} ms]{RESET}")

        message = response.get("message") or {}
        content = str(message.get("content") or "")
        calls = message.get("tool_calls") or []

        # We never offered tools on a no-tools round, so anything shaped like a tool call
        # here is spurious. Clearing it guarantees such a round can only terminate.
        if not use_tools:
            calls = []

        if not calls:
            # Gemma answered without calling a tool while still believing this is a tool
            # task. The router gets exactly ONE shot at naming the kind — never on a
            # synthesis round, never twice, and never over a real decision.
            if tool_mode and use_tools and not final_synthesis and not rescued:
                routed = needle_route(user_text, needle)
                if routed:
                    rescued = True
                    println(f"{DIM}↳ router picked {routed}{RESET}")
                    calls = [{
                        "function": {
                            "name": "tool_wielder",
                            "arguments": {
                                "kind": routed,
                                "instruction": clip(user_text, 2000),
                            },
                        }
                    }]
            if not calls:
                answer = content.strip() or "The Main Brain returned an empty response."
                history.append({"role": "assistant", "content": content})
                return answer

        history.append({"role": "assistant", "content": content, "tool_calls": calls})

        # Gemma authored the script; Needle completes it into concrete arguments.
        grounded = needle_ground(calls, needle)
        if grounded is not None:
            calls = grounded

        planned = _plan_tool_calls(calls, seen_calls, prior_call_args)
        _execute_planned(planned, worker, qwen, LOCAL_EXEC)

        # Qwen's review is the slow remote brain; reduction and verification are local.
        # Start the review on a thread and do every local step underneath it, then join
        # before history is touched — review_planned mutates each result's "review" key,
        # and compact_tool_result embeds that key, so writes must land before message
        # building. Extraction only ever reads result fields for its own prompt, so it is
        # safe to run while the thread writes reviews; a raced read is at worst a prompt
        # missing an attribute, never a corrupt result.
        review_thread: Optional[threading.Thread] = None
        if QWEN_REVIEW and qwen is not None and not force_no_tools_next:
            review_thread = threading.Thread(
                target=review_planned, args=(planned, qwen), daemon=True
            )
            review_started = time.monotonic()
            review_thread.start()

        round_had_failure = False
        round_had_success = False
        round_had_duplicate = False
        round_labels: List[str] = []
        round_results: List[Dict[str, Any]] = []
        reductions: List[Optional[str]] = []

        # Local reductions overlap the remote review. Nothing here writes history. Wrapped in
        # try/finally so the review thread is ALWAYS joined: if a reduction raised, a daemon
        # thread still writing result["review"] while history is built would be a data race.
        try:
            for item in planned:
                result = item.get("result") or {
                    "ok": False, "kind": item.get("kind"), "error": "no result produced",
                }
                args = item.get("args") or {}
                label = str(
                    args.get("instruction") or args.get("path") or args.get("query")
                    or args.get("url") or ""
                ).strip()

                if item["duplicate"]:
                    round_had_duplicate = True
                elif result.get("ok"):
                    round_had_success = True
                    if item["needs_worker"] and item["key"] is not None:
                        seen_calls[item["key"]] = dict(result)
                else:
                    round_had_failure = True

                if item.get("needs_worker") and isinstance(item.get("result"), dict):
                    if label:
                        round_labels.append(label)
                    round_results.append(item["result"])

                # Anything big enough to matter is reduced by the extractor first, so Gemma
                # spends its window on the facts rather than on the raw dump — without ever
                # losing the envelope or Qwen's verdict on it.
                reductions.append(extract_tool_result(result, label, needle) if label else None)
        finally:
            if review_thread is not None:
                review_thread.join()
                if SHOW_TIMINGS:
                    review_ms = round((time.monotonic() - review_started) * 1000)
                    println(f"{DIM}[Qwen review {review_ms} ms, overlapped]{RESET}")

        # Reviews are now attached; build every tool message so each one embeds its verdict.
        for item, reduced in zip(planned, reductions):
            result = item.get("result") or {
                "ok": False, "kind": item.get("kind"), "error": "no result produced",
            }
            args = item.get("args") or {}
            label = str(
                args.get("instruction") or args.get("path") or args.get("query")
                or args.get("url") or ""
            ).strip()

            tool_message: Dict[str, Any] = {
                "role": "tool",
                "content": compact_tool_result(result, reduced=reduced),
            }
            if item["name"]:
                tool_message["tool_name"] = item["name"]
            history.append(tool_message)
            last_round_tools.append(tool_message["content"])

        # One verdict per round, hung off the last tool message so Gemma reads the work
        # and then immediately reads whether that work was actually enough. When it was
        # not, Needle drafts ONE concrete retry proposal into the same message — Gemma's
        # next round is then a short "approve / adjust" instead of a from-scratch re-plan.
        if round_results:
            verification = needle_verify(
                "; ".join(round_labels) or user_text, round_results, needle
            )
            if verification and history and history[-1].get("role") == "tool":
                history[-1]["verify"] = verification
                if not verification["satisfied"] and verification["gap"]:
                    last_item = planned[-1]
                    last_result = last_item.get("result") or {}
                    last_args = last_item.get("args") or {}
                    fix = needle_draft_fix(
                        str(
                            last_args.get("instruction")
                            or last_args.get("path")
                            or last_args.get("query")
                            or last_args.get("url")
                            or user_text
                        ),
                        str(last_args.get("kind") or last_result.get("kind") or "execute"),
                        verification["gap"],
                        needle,
                    )
                    if fix:
                        history[-1]["draft_fix"] = fix

        # A duplicate-only round means the model already holds the answer: stop working
        # and spend one pass turning what it has into a response.
        if round_had_duplicate and not round_had_failure and not round_had_success:
            force_no_tools_next = True
            final_synthesis = True
            round_no += 1
            continue

        # Genuine progress resets the stall detector. Repeated failing/no-op rounds trigger
        # a final synthesis pass rather than a misleading "maximum rounds" error.
        if round_had_success:
            stalled_rounds = 0
        elif round_had_failure or round_had_duplicate:
            stalled_rounds += 1
        else:
            stalled_rounds = 0

        if stalled_rounds >= max(1, MAX_STALL_ROUNDS):
            force_no_tools_next = True
            final_synthesis = True
        else:
            force_no_tools_next = False

        # Mid-task compaction (opt-in): a long multi-round task piles up tool messages while
        # OLD completed turns sit untouched. When the session has outgrown the window, distill
        # them now so every remaining round carries the short memory note instead of the bulk.
        # Slice-assignment keeps the caller's list object in sync (run_turn mutates in place).
        if COMPACT_MID_TASK and brain is not None and len(_group_turns(history)) > MAX_HISTORY:
            history[:] = maybe_compact_history(history, brain)

        tool_mode = True
        round_no += 1

    return ("The task loop hit its round ceiling before completion; "
            "raise MAX_ROUNDS_HARD or MAX_TOOL_ROUNDS to allow more steps.")

# ============================================================
# PREWARM
# ============================================================

def warm_main_background(brain: MainOllama) -> None:
    try:
        loaded = False
        try:
            loaded = any(
                str(item.get("name", "")) == MAIN_MODEL
                for item in brain.ps().get("models", [])
            )
        except Exception:
            pass
        if loaded:
            println(f"{GREEN}✓ Main Brain already warm{RESET}")
            return
        println(f"{DIM}↳ prewarming Main Brain...{RESET}")
        started = time.monotonic()
        brain.warm()
        println(f"{GREEN}✓ Main Brain warm ({time.monotonic() - started:.1f}s){RESET}")
    except Exception as exc:
        println(f"{YELLOW}⚠ Main prewarm skipped: {exc}{RESET}")


def warm_worker_background(worker: ToolClient) -> None:
    try:
        println(f"{DIM}↳ prewarming Tool Brain...{RESET}")
        started = time.monotonic()
        data = worker.warm()
        elapsed = time.monotonic() - started
        if data.get("ok"):
            println(f"{GREEN}✓ Tool Brain warm ({elapsed:.1f}s){RESET}")
        else:
            println(f"{YELLOW}⚠ Tool Brain prewarm failed: {compact_json(data)}{RESET}")
    except Exception as exc:
        println(f"{YELLOW}⚠ Tool Brain prewarm skipped: {exc}{RESET}")


def warm_needle_background(needle: Optional["NeedleRouter"]) -> None:
    """Probe every local Needle role once; drop the ones that refuse the connection.

    The shim is optional, but a stopped shim used to surface as "needle:planner degraded"
    in the middle of a turn, which reads like the agent broke. One honest line at startup —
    and skipping the dead roles — is the better story.
    """
    if needle is None or not needle.active:
        return
    down: List[str] = []
    url = ""
    for role, up in needle.health().items():
        if not up:
            down.append(role)
            if not url:
                url = str((needle.slots.get(role) or {}).get("url") or "")
    for role in down:
        needle.clients.pop(role, None)
    if down:
        println(
            f"{YELLOW}⚠ Needle {'/'.join(down)} unreachable at {url or NEEDLE_HOST}"
            f" — start the local shim (start-needle.sh); running without "
            f"{'them' if len(down) > 1 else 'it'}.{RESET}"
        )


def prewarm_all(brain: MainOllama, worker: ToolClient,
                needle: Optional["NeedleRouter"] = None) -> None:
    """Start optional prewarming without ever delaying the interactive prompt."""
    threading.Thread(target=warm_main_background, args=(brain,), daemon=True).start()
    threading.Thread(target=warm_worker_background, args=(worker,), daemon=True).start()
    if needle is not None:
        threading.Thread(target=warm_needle_background, args=(needle,), daemon=True).start()


def warm_qwen_background(qwen: QwenOllama) -> None:
    try:
        started = time.monotonic()
        qwen.warm()
        println(f"{GREEN}✓ Tool Brain warm ({time.monotonic() - started:.1f}s){RESET}")
    except Exception as exc:
        println(f"{YELLOW}⚠ Tool Brain prewarm skipped: {exc}{RESET}")


# ============================================================
# WORKER MODE
# ============================================================

def worker_mode() -> None:
    if not TOOL_WIELDER_KEY:
        raise SystemExit("TOOL_WIELDER_KEY is required for worker mode")

    ensure_memory_dir()
    load_memory_cache()
    qwen = QwenOllama()

    println(f"Model:     {QWEN_MODEL}")
    println(f"Workspace: {WORKSPACE}")
    println(f"Memory:    {MEMORY_FILE}")
    println(f"Memories:  {len(_memory_cache)}")
    println(f"Listen:    {os.getenv('WORKER_LISTEN_HOST', '0.0.0.0')}:{TOOL_PORT}")

    server = BraincellsHTTPServer(
        (os.getenv("WORKER_LISTEN_HOST", "0.0.0.0"), TOOL_PORT),
        WorkerHandler,
    )
    server.qwen = qwen

    if PREWARM:
        threading.Thread(
            target=lambda: warm_qwen_background(qwen),
            daemon=True,
        ).start()

    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        qwen.close()


# ============================================================
# MAIN MODE
# ============================================================

def health_main(brain: MainOllama, worker: ToolClient,
                needle: Optional[NeedleRouter] = None) -> None:
    try:
        data = brain.tags()
        names = [str(item.get("name")) for item in data.get("models", [])]
        println(f"{GREEN}✓ Main Brain reachable{RESET} ({MAIN_MODEL}: {'yes' if MAIN_MODEL in names else 'not listed'})")
    except Exception as exc:
        eprintln(f"Main Brain: {exc}")

    try:
        data = worker.health()
        println(f"{GREEN}✓ Tool Wielder reachable{RESET} {compact_json(data)}")
    except Exception as exc:
        eprintln(f"Tool Wielder: {exc}")

    if needle is None or not needle.active:
        println(f"{DIM}✓ Needle: no active roles (v4 behaviour){RESET}")
        return
    for role, up in needle.health().items():
        mark = f"{GREEN}✓" if up else f"{RED}✗"
        println(f"{mark} Needle {role} {'reachable' if up else 'unreachable'}{RESET} "
                f"{DIM}{needle.slots[role].get('url')}{RESET}")


def main_cli() -> None:
    global SHOW_TIMINGS
    # The key only guards the REMOTE Tool Wielder. With local execution there is no remote
    # endpoint to authenticate to, so demanding a key would block a perfectly valid local run.
    if not TOOL_WIELDER_KEY and not LOCAL_EXEC:
        raise SystemExit(
            "TOOL_WIELDER_KEY is required (remote tool execution).\n"
            "Run: export TOOL_WIELDER_KEY='your-key'\n"
            "Or run locally with --local / NEEDLE_LOCAL_EXEC=1."
        )

    println(f"{BOLD}{MAGENTA}╔══════════════════════════════════════════╗")
    println("║      MR. BRAINCELLS — THREE BRAINS       ║")
    println(f"╚══════════════════════════════════════════╝{RESET}")

    brain = MainOllama()
    worker = ToolClient(TOOL_HOST, TOOL_PORT, TOOL_WIELDER_KEY, max(MAIN_TIMEOUT, 120))
    needle = build_needle_router()
    qwen = RemoteQwen(worker) if QWEN_REVIEW else None
    active = needle.active

    println(
        f"{DIM}CEO    {MAIN_MODEL} @ {MAIN_HOST}          plans, scripts, decides, answers\n"
        f"REVIEW {QWEN_MODEL} @ {TOOL_HOST}:{TOOL_PORT}        reads every result, "
        f"reports status and fixes\n"
        f"NEEDLE {len(active)}/5 roles @ "
        f"{', '.join(active) if active else 'none'}  "
        f"executes Gemma's script on this box\n"
        f"{RESET}"
    )
    println(
        f"{DIM}exec={'local' if LOCAL_EXEC else 'remote'}  "
        f"review={'on' if QWEN_REVIEW else 'off'}  "
        f"conf>={NEEDLE_CONFIDENCE:.2f}  {len(NEEDLE_ROLES)} named roles "
        f"({', '.join(NEEDLE_ROLES)}){RESET}\n"
    )
    if not active:
        println(f"{YELLOW}⚠ Needle has no active roles — Gemma + Qwen only "
                f"(no scope/route/ground/verify/extract). "
                f"Run --needle-setup to light up all five at once.{RESET}\n")
    # The single biggest source of "it didn't make my file!" is expecting the tools to
    # touch this box while they are actually running on the remote worker. Say it plainly.
    if LOCAL_EXEC:
        println(f"{DIM}tools run HERE, in {WORKSPACE} (Gemma's script executes locally).{RESET}\n")
    else:
        println(f"{YELLOW}⚠ tools run on the REMOTE Tool Wielder ({TOOL_HOST}:{TOOL_PORT}) — "
                f"files and commands land there, not here. Pass --local to run them on "
                f"this machine.{RESET}\n")

    try:
        # Startup is deliberately non-blocking: no network request or model generation may delay the prompt.
        history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        println(f"{DIM}/health  /reset  /config  /needle  /warm  /help  /quit{RESET}")
        println(f"{GREEN}✓ CLI ready{RESET}")

        def startup_background() -> None:
            # Absolutely no startup network/model work unless explicitly requested.
            if PREWARM:
                prewarm_all(brain, worker, needle)

        if PREWARM:
            threading.Thread(target=startup_background, daemon=True).start()
        println()

        while True:
            try:
                user = input(f"{BOLD}{CYAN}you › {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                println()
                break

            if not user:
                continue

            if user in {"/quit", "/exit"}:
                break

            if user == "/help":
                println(f"{DIM}/health /reset /config /needle /memories /clear-memories "
                        f"/warm /timings /help /quit{RESET}")
                continue

            if user == "/reset":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                _session_memory.clear()
                _review_cache_reset()
                _breaker_reset()
                println(f"{GREEN}✓ conversation reset{RESET}")
                continue

            if user == "/config":
                println(
                    f"Main : {MAIN_MODEL} @ {MAIN_HOST}\n"
                    f"Tool : {QWEN_MODEL} @ {TOOL_HOST}:{TOOL_PORT}\n"
                    f"ctx={MAIN_CTX} chat_ctx={CHAT_CTX} output={MAIN_MAX_OUTPUT} keep_alive={MAIN_KEEP_ALIVE}\n"
                    f"temp={MAIN_TEMPERATURE}/{SYNTH_TEMPERATURE} chat_output={CHAT_MAX_OUTPUT} "
                    f"rounds={'adaptive' if MAX_TOOL_ROUNDS <= 0 else MAX_TOOL_ROUNDS} "
                    f"stall_limit={MAX_STALL_ROUNDS} hard_ceiling={MAX_ROUNDS_HARD}\n"
                    f"history_turns={TOOL_HISTORY_TURNS} prewarm={PREWARM} timings={SHOW_TIMINGS}\n"
                    f"stream={STREAM} tool_parallel={TOOL_PARALLEL} conn_pool={TOOL_CONN_POOL} "
                    f"search_parallel={SEARCH_PARALLEL}\n"
                    f"connect_timeout={HTTP_CONNECT_TIMEOUT:.0f}s breaker(req {BREAKER_THRESHOLD} / "
                    f"{BREAKER_COOLDOWN}s) review_cache={'on' if REVIEW_CACHE else 'off'}\n"
                    f"compact={'on' if COMPACT_MEMORY else 'off'} max={COMPACT_MEMORY_MAX} "
                    f"keep={COMPACT_MEMORY_KEEP} ctx={COMPACT_CONTEXT_MAX} "
                    f"draft_fix={'on' if DRAFT_FIX else 'off'}\n"
                    f"exec={'local' if LOCAL_EXEC else 'remote'} review={'on' if QWEN_REVIEW else 'off'} "
                    f"needleroles={'/'.join(NEEDLE_ROLES)} "
                    f"active={','.join(needle.active) or 'none'} "
                    f"conf>={NEEDLE_CONFIDENCE:.2f} "
                    f"extract>={NEEDLE_EXTRACT_MIN}"
                )
                continue

            if user == "/needle":
                print_needle_slots()
                for role in NEEDLE_ROLES:
                    client = needle.clients.get(role)
                    if client is None:
                        continue
                    up = client.health()
                    println(f"{DIM}  probe {role}: "
                            f"{'up' if up else 'down'}{RESET}")
                continue

            if user == "/health":
                health_main(brain, worker, needle)
                continue

            if user == "/memories":
                try:
                    data = worker.memories()
                    rows = data.get("memories", [])
                    if not rows:
                        println(f"{DIM}No stored memories.{RESET}")
                    else:
                        for row in rows:
                            println(f"{CYAN}{row.get('id', '?')}{RESET}  {row.get('content', '')}")
                except Exception as exc:
                    eprintln(f"Memories: {exc}")
                continue

            if user == "/clear-memories":
                try:
                    data = worker.clear_memories()
                    println(f"{GREEN}✓ cleared {data.get('removed', 0)} memories{RESET}")
                except Exception as exc:
                    eprintln(f"Clear memories: {exc}")
                continue

            if user == "/timings":
                SHOW_TIMINGS = not SHOW_TIMINGS
                println(f"{GREEN}✓ timings {'ON' if SHOW_TIMINGS else 'OFF'}{RESET}")
                continue

            if user == "/warm":
                prewarm_all(brain, worker, needle)
                println(f"{GREEN}✓ background warm started{RESET}")
                continue

            started = time.monotonic()

            # Tokens are printed as they arrive instead of after the whole generation, so
            # a long final answer starts showing up immediately rather than after a long
            # silent pause. The header is emitted lazily on the first delta so it always
            # lands after any "↳ tool" lines from earlier rounds in the same turn.
            started_header = {"done": False}

            def on_delta(piece: str) -> None:
                with _print_lock:
                    if not started_header["done"]:
                        started_header["done"] = True
                        print(f"\n{BOLD}{GREEN}braincells › {RESET}", end="", flush=True)
                    print(piece, end="", flush=True)

            answer = run_turn(
                user, history, brain, worker, on_delta=on_delta,
                needle=needle, qwen=qwen,
            )
            total = time.monotonic() - started

            if started_header["done"]:
                println()
            else:
                println(f"\n{BOLD}{GREEN}braincells › {RESET}{answer}")
            println(f"{DIM}[total {total:.2f}s]{RESET}\n")

            # Trim by COMPLETE user turns, never by individual tool messages. This preserves
            # continuation context so “continue” can actually resume an incomplete task.
            # When something has to go, the CEO (Gemma) first writes the one note about the
            # dropped turns worth keeping; a cheap mechanical trim then lets it slide.
            if len(_group_turns(history)) > MAX_HISTORY and COMPACT_MEMORY:
                history = maybe_compact_history(history, brain)
            elif len(_group_turns(history)) > MAX_HISTORY:
                history = trim_history_turns(history, MAX_HISTORY)
    finally:
        brain.close()
        worker.close()
        needle.close()
        if qwen:
            qwen.close()
        println(f"{DIM}Mr. Braincells offline.{RESET}")


# ============================================================
# BOTH MODE
# ============================================================

def both_mode() -> None:
    """Three-brain client: Gemma plans, Qwen reviews, Needle runs the script."""
    main_cli()


# ============================================================
# ENTRYPOINT
# ============================================================

# ============================================================
# NEEDLE CONFIG (the --needle-* commands)
# ============================================================

def _needle_slot_line(role: str, slot: Dict[str, Any], active: set) -> str:
    on = role in active
    mark = f"{GREEN}on {RESET}" if on else f"{DIM}off{RESET}"
    key = "key=yes" if slot.get("key") else "key=no"
    thr = needle_threshold(slot)
    return (
        f"{BOLD}{role:<10}{RESET} {mark}  "
        f"{slot.get('url') or '(unset)'}  model={slot.get('model')}  "
        f"threshold={thr:.2f} format={slot.get('format')} {key}"
    )


def print_needle_slots(slots: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    src = slots if slots is not None else NEEDLE_SLOTS
    active = set(needle_active_roles(src))
    println(f"{BOLD}NEEDLE BRAIN{RESET} {DIM}(local executioner — Gemma writes the script, "
            f"Needle runs it){RESET}")
    for role in NEEDLE_ROLES:
        println(_needle_slot_line(role, src[role], active))
        println(f"{DIM}{'':<10} {NEEDLE_ROLE_BRIEF[role]}{RESET}")
    println(f"{DIM}active={len(active)}/5  "
            f"local_exec={'on' if compute_local_exec(src) else 'off'}  "
            f"review={'on' if QWEN_REVIEW else 'off'}  host={NEEDLE_HOST}{RESET}")


def needle_cli_command(args: argparse.Namespace) -> bool:
    """Handle every --needle-* flag. Returns True when the CLI should stop here.

    Configuration is a one-shot: it writes config.json and exits, so no model, server
    or worker is ever started just to change an endpoint.
    """
    slots = NEEDLE_SLOTS

    if getattr(args, "needle_list", False):
        print_needle_slots(slots)
        return True

    if getattr(args, "needle_probe", False):
        router = build_needle_router()
        if not router.active:
            println(f"{YELLOW}No active Needle roles. Use --needle-setup.{RESET}")
        for role in NEEDLE_ROLES:
            client = router.clients.get(role)
            if client is None:
                println(f"{role:<10} {DIM}inactive{RESET}")
                continue
            up = client.health()
            verdict = f"{GREEN}up{RESET}" if up else f"{RED}down{RESET}"
            println(f"{role:<10} {verdict}  {client.http.host}:{client.http.port}")
        router.close()
        return True

    if getattr(args, "needle_setup", False):
        # One shot for the whole squad: every role gets the same endpoint, so a single
        # running Needle turns the CLI from 0/5 to 5/5 without five separate commands.
        url = (getattr(args, "needle_url", None) or NEEDLE_HOST).strip()
        try:
            _split_needle_url(url)
        except ValueError as exc:
            raise SystemExit(str(exc))
        model = (getattr(args, "needle_model", None) or "").strip() or "needle-2"
        fmt = getattr(args, "needle_format", None) or "openai"
        threshold = getattr(args, "needle_threshold", None)
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise SystemExit(
                "--needle-threshold must be between 0 and 1 (got %s); "
                "Needle reports confidence on a 0..1 scale" % threshold
            )
        for role in NEEDLE_ROLES:
            slot = slots[role]
            slot["url"] = url
            slot["model"] = model
            slot["format"] = fmt
            slot["enabled"] = True
            if getattr(args, "needle_api_key", None) is not None:
                slot["key"] = args.needle_api_key
            if threshold is not None:
                slot["threshold"] = float(threshold)
        save_needle_slots(slots)
        println(f"{GREEN}✓ Needle squad configured: all {len(NEEDLE_ROLES)} roles -> "
                f"{url} (model={model}, format={fmt}){RESET}")
        print_needle_slots(slots)
        println(f"{DIM}tip: start your Needle server, then --needle-probe to check it. "
                f"Set NEEDLE_LOCAL_EXEC=0 to keep tool calls on the remote worker.{RESET}")
        return True

    clear_role = getattr(args, "needle_clear", None)
    if clear_role:
        slots[clear_role] = dict(NEEDLE_SLOT_DEFAULT, url="", enabled=False)
        save_needle_slots(slots)
        println(f"{GREEN}✓ cleared Needle role '{clear_role}'{RESET}")
        print_needle_slots(slots)
        return True

    for flag, want_on in (("needle_on", True), ("needle_off", False)):
        role = getattr(args, flag, None)
        if not role:
            continue
        slot = slots[role]
        slot["enabled"] = want_on
        if want_on and not slot.get("url"):
            slot["url"] = NEEDLE_HOST
        save_needle_slots(slots)
        state = "enabled" if want_on else "disabled"
        println(f"{GREEN}✓ {state} Needle role '{role}' -> "
                f"{slot.get('url') or '(unset)'}{RESET}")
        return True

    set_role = getattr(args, "needle_set", None)
    if not set_role:
        return False

    slot = slots[set_role]
    changed = []
    if getattr(args, "needle_url", None) is not None:
        slot["url"] = args.needle_url.strip()
        changed.append("url")
    if getattr(args, "needle_api_key", None) is not None:
        slot["key"] = args.needle_api_key
        changed.append("key")
    if getattr(args, "needle_model", None) is not None:
        slot["model"] = args.needle_model.strip()
        changed.append("model")
    if getattr(args, "needle_threshold", None) is not None:
        value = float(args.needle_threshold)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(
                "--needle-threshold must be between 0 and 1 (got %s); "
                "Needle reports confidence on a 0..1 scale" % args.needle_threshold
            )
        slot["threshold"] = value
        changed.append("threshold")
    if getattr(args, "needle_format", None) is not None:
        slot["format"] = args.needle_format
        changed.append("format")
    if not changed:
        raise SystemExit(
            f"--needle-set {set_role} needs at least one of "
            "--needle-url / --needle-api-key / --needle-model / --needle-threshold / --needle-format"
        )
    if not slot.get("url"):
        slot["url"] = NEEDLE_HOST
    try:
        _split_needle_url(slot["url"])
    except ValueError as exc:
        raise SystemExit(str(exc))
    slot["enabled"] = True

    save_needle_slots(slots)
    println(f"{GREEN}✓ Needle role '{set_role}' configured "
            f"({', '.join(changed)}){RESET}")
    # Never echo the key back, on screen or in a shell history.
    print_needle_slots(slots)
    println(f"{DIM}tip: --needle-probe checks every active endpoint{RESET}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mr. Braincells Type 2 — three-brain CLI (Gemma CEO, Needle "
                    "local executor, Qwen reviewer)"
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="run only the Tool Wielder HTTP service",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="run the three-brain client using remote Main + remote Tool Wielder",
    )
    parser.add_argument(
        "--key",
        metavar="KEY",
        help="override TOOL_WIELDER_KEY for this run; use --key - to enter it privately",
    )
    parser.add_argument(
        "--ip",
        action="store_true",
        help="interactively choose Main or Worker and save its new IPv4 address",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        dest="exec_local",
        help="run Gemma's tool calls on THIS machine in the local workspace "
             "(default when a Needle role is active or NEEDLE_LOCAL_EXEC=1)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        dest="exec_remote",
        help="send Gemma's tool calls to the remote Tool Wielder instead of running here",
    )

    needle = parser.add_argument_group(
        "needle brain (local executioner)",
        "Gemma writes the script, Needle runs it on this device, Qwen reviews every "
        "output and reports back to Gemma. Configure one endpoint per role.",
    )
    needle.add_argument(
        "--needle-set",
        metavar="ROLE",
        choices=list(NEEDLE_ROLES),
        help=f"configure a role ({', '.join(NEEDLE_ROLES)}) and save it",
    )
    needle.add_argument(
        "--needle-setup",
        action="store_true",
        help="one-shot: point ALL five roles at a single endpoint and enable them "
             "(defaults to NEEDLE_HOST; override with --url/--model/--needle-api-key)",
    )
    needle.add_argument("--needle-url", "--url", metavar="URL",
                        help="endpoint, e.g. http://127.0.0.1:8000")
    needle.add_argument("--needle-api-key", metavar="KEY",
                        help="bearer token for that endpoint")
    needle.add_argument("--needle-model", "--model", metavar="NAME",
                        help="model id sent to that endpoint")
    needle.add_argument(
        "--needle-threshold", "--threshold",
        metavar="0..1",
        type=float,
        help="minimum confidence before Needle is allowed to act (default "
        f"{NEEDLE_CONFIDENCE})",
    )
    needle.add_argument(
        "--needle-format", "--format",
        choices=["openai", "native", "auto"],
        help="wire format of the endpoint (default auto-detect)",
    )
    needle.add_argument("--needle-list", action="store_true", help="show all five roles")
    needle.add_argument("--needle-probe", action="store_true", help="health-check every active role")
    needle.add_argument("--needle-clear", metavar="ROLE", choices=list(NEEDLE_ROLES),
                        help="forget a role's endpoint")
    needle.add_argument("--needle-on", metavar="ROLE", choices=list(NEEDLE_ROLES),
                        help="enable a role")
    needle.add_argument("--needle-off", metavar="ROLE", choices=list(NEEDLE_ROLES),
                        help="disable a role without forgetting its endpoint")
    return parser.parse_args()


def main() -> None:
    global LOCAL_EXEC
    args = parse_args()
    if args.worker and args.both:
        raise SystemExit("Choose --worker or --both, not both.")
    if args.exec_local and args.exec_remote:
        raise SystemExit("Choose --local or --remote, not both.")

    apply_cli_overrides(args)

    # Where Gemma's script runs. --local/--remote win over the config-derived default so a
    # user on their own machine can point "write this file here" at their real directory
    # without having to stand up a Needle endpoint first.
    if args.exec_local:
        LOCAL_EXEC = True
    elif args.exec_remote:
        LOCAL_EXEC = False

    # Config flags are pure and one-shot: they never start a server, worker or model.
    if needle_cli_command(args):
        return

    try:
        if args.worker:
            worker_mode()
        elif args.both:
            both_mode()
        else:
            main_cli()
    except KeyboardInterrupt:
        # Ctrl+C while a model is streaming is a "stop", not a crash: exit quietly.
        println(f"\n{DIM}Interrupted.{RESET}")
    except Exception as exc:
        # A raw traceback for a one-line misconfiguration is hostile; keep it behind a flag.
        if os.getenv("BRAIN_DEBUG", "0").lower() in {"1", "true", "yes"}:
            raise
        eprintln(f"{RED}Error: {type(exc).__name__}: {exc}{RESET}")
        eprintln(f"{DIM}Re-run with BRAIN_DEBUG=1 for the full traceback.{RESET}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
