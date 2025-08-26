import os
import signal
import subprocess
import sys
import time
import json
import atexit
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import functools
import requests
import streamlit as st
import copy
import re
from pathlib import Path


# ============================
# Load .env BEFORE reading env vars
# ============================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================
# Streamlit configuration
# ============================
st.set_page_config(page_title="TP LLM: Ollama · llama.cpp · ChatGPT API", page_icon="", layout="wide")

TITLE = "TP LLM"
st.title(TITLE)
st.caption("Run two independent instances side-by-side to compare outputs in real time. Each side can be: Remote (Ollama or llama.cpp), Host a local llama.cpp server, or ChatGPT API. Tools (Web Search, File Search, Code Interpreter) are supported on the ChatGPT API side.")

# ============================
# Constants
# ============================
MODELS_DIR = Path("models")
DEFAULT_THREADS = int(os.environ.get("LLAMA_THREADS", "8"))
SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")  # llama.cpp server binary

HOST_BIND = os.environ.get("LLAMA_HOST", "0.0.0.0")       # bind address for hosted llama.cpp server
CLIENT_HOST = os.environ.get("LLAMA_CLIENT", "127.0.0.1")  # how we reach local servers

DEFAULT_PORT_A = int(os.environ.get("LLAMA_PORT_A", "8000"))
DEFAULT_PORT_B = int(os.environ.get("LLAMA_PORT_B", "8001"))

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", f"http://127.0.0.1:{DEFAULT_PORT_A}")

# Suggested Ollama models (include GPT-OSS)
SUGGESTED_OLLAMA_MODELS = [
    "gpt-oss:20b",
    "gpt-oss:120b",
    "llama3:8b",
    "llama3:70b",
    "qwen2.5:7b",
    "mistral:7b",
]

# OpenAI model lists (Responses API capable; primary text/chat models)
OPENAI_CORE_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-chat-latest"]
OPENAI_MORE_MODELS = ["o3", "o3-mini", "o3-pro", "gpt-4.1", "gpt-4o"]

if "last_output" not in st.session_state:
    st.session_state["last_output"]= {}
if "last_thinking" not in st.session_state:
    st.session_state.last_thinking = {"A": None, "B": None}

# ============================
# Session defaults per side
# ============================

def _default_side(name: str, default_port: int) -> Dict[str, Any]:
    return {
        "name": name,
        "mode": "Remote Server",              # Remote Server | Host llama.cpp | ChatGPT API
        "remote_kind": "Ollama",              # if Remote Server: Ollama | llama.cpp
        # ---- Remote: Ollama ----
        "ollama": {
            "url": DEFAULT_OLLAMA_URL,
            "model": "gpt-oss:20b",
            "models_from_server": [],
            "native_api": True,  # /api/chat; if False, use OpenAI-compatible /v1/chat/completions
            "keep_alive": "5m",
            "sampling": {
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 40,
                "min_p": 0.05,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "tfs_z": 1.0,
                "mirostat": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "seed": 0,
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        },
        # ---- Remote: llama.cpp ----
        "llamacpp_remote": {
            "url": DEFAULT_LLAMA_CPP_URL,
            "sampling": {
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 40,
                "min_p": 0.05,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "tfs_z": 1.0,
                "mirostat": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "seed": 0,
            },
            "max_tokens": 4096,
        },
        # ---- Host llama.cpp ----
        "llamacpp_host": {
            "proc": None,
            "port": default_port,
            "threads": DEFAULT_THREADS,
            "model_path": None,
            # reuse remote sampling settings UI when hosting
        },
        # ---- OpenAI (ChatGPT API) ----
        "openai": {
            "api_base": "https://api.openai.com",
            "api_key": os.environ.get("OPENAI_API_KEY") or None,
            "model": "gpt-5",
            "sampling": {
                "temperature": 0.0,
                "top_p": 0.95,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
            "verbosity": "medium",
            "reasoning_effort": "medium",
            "max_output_tokens": 4096,

            # 🔧 NEW: Tool config
            "tools": {
                "web_search": True,          # out-of-the-box
                "file_search": False,        # requires vector_store_ids
                "code_interpreter": False,   # OpenAI sandboxed Python
            },
            "file_search": {
                "vector_store_ids": [],      # fill with your vector store IDs to enable File Search
            },
            # "auto" | "none" | "web_search" | "file_search" | "code_interpreter"
            "tool_choice": "auto",
        },
    }

if "sideA" not in st.session_state:
    st.session_state.sideA = _default_side("A", DEFAULT_PORT_A)
if "sideB" not in st.session_state:
    st.session_state.sideB = _default_side("B", DEFAULT_PORT_B)

# ============================
# Helpers
# ============================
class ServerLaunchError(Exception):
    pass

class StreamError(Exception):
    pass

def list_local_gguf(models_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if models_dir.exists():
        for p in models_dir.rglob("*.gguf"):
            out[str(p.relative_to(models_dir))] = p.resolve()
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _pgid_kill(proc: subprocess.Popen):
    if not proc:
        return
    try:
        if sys.platform.startswith("win"):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
    except Exception:
        pass


def stop_host_llamacpp(side_key: str):
    info = st.session_state[side_key]["llamacpp_host"]
    proc = info.get("proc")
    if proc and getattr(proc, "poll", lambda: None)() is None:
        _pgid_kill(proc)
    info["proc"] = None


def cleanup_on_exit():
    stop_host_llamacpp("sideA")
    stop_host_llamacpp("sideB")

atexit.register(cleanup_on_exit)
try:
    def _handle_exit_signal(signum, frame):
        cleanup_on_exit()
        try:
            sys.exit(0)
        except SystemExit:
            raise
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _handle_exit_signal)
except Exception:
    pass


def start_host_llamacpp(side_key: str, model_path: Path, port: int, threads: int = DEFAULT_THREADS) -> None:
    """Start (or restart) a local llama.cpp server for a side.
    Uses --gpu-layers 0 to force full CPU by default.
    """
    stop_host_llamacpp(side_key)

    cmd = [
        SERVER_BIN,
        "-m", str(model_path),
        "--port", str(port),
        "--host", HOST_BIND,
        "-t", str(threads),
        "--gpu-layers", "0",
        "-c", "0",
        "--reasoning-format", "none",
        "--jinja",
        "-fa",
    ]

    creationflags = 0
    preexec_fn = None
    if not sys.platform.startswith("win"):
        preexec_fn = os.setsid
    else:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=preexec_fn,
            creationflags=creationflags,
        )
    except FileNotFoundError as e:
        raise ServerLaunchError(
            f"Cannot find '{SERVER_BIN}'. Put it in PATH or set LLAMA_SERVER_BIN."
        ) from e
    except Exception as e:
        if proc and proc.poll() is None:
            _pgid_kill(proc)
        raise ServerLaunchError(str(e)) from e

    host_info = st.session_state[side_key]["llamacpp_host"]
    host_info.update({
        "proc": proc,
        "port": int(port),
        "threads": int(threads),
        "model_path": str(model_path),
    })

    base = f"http://{CLIENT_HOST}:{port}"
    deadline = time.time() + 30
    last_err = None
    while time.time() < deadline:
        for path in ("/health", "/healthz", "/"):
            try:
                r = requests.get(base + path, timeout=1)
                if r.status_code < 500:
                    return
            except Exception as e:
                last_err = e
        time.sleep(0.5)

    if proc and proc.poll() is None:
        _pgid_kill(proc)
    raise ServerLaunchError(f"Local llama.cpp for side {side_key[-1]} did not become ready on {base}. Last error: {last_err}")


# -------- Streaming helpers --------

def _headers_json(api_key: Optional[str] = None) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def parse_sse_lines(iter_lines):
    for raw in iter_lines:
        if raw is None:
            continue
        line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):].strip()
        else:
            payload = line
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except Exception:
            continue


def extract_delta_text(obj: Dict[str, Any]) -> str:
    """Extract streamed delta text from multiple backends.
    - OpenAI / llama.cpp (OpenAI-compatible): choices[0].delta.content or choices[0].text
    - Ollama native /api/chat: message.content (or delta)
    - Ollama /api/generate: response
    """
    # OpenAI / llama.cpp OpenAI-compatible
    try:
        choices = obj.get("choices") or []
        if choices:
            c0 = choices[0]
            delta = c0.get("delta") or {}
            if isinstance(delta, dict):
                dc = delta.get("content")
                if isinstance(dc, str) and dc:
                    return dc
            # some servers stream as .text
            txt = c0.get("text")
            if isinstance(txt, str) and txt:
                return txt
    except Exception:
        pass

    # Ollama native /api/chat
    try:
        msg = obj.get("message")
        if isinstance(msg, dict):
            mc = msg.get("content")
            if isinstance(mc, str) and mc:
                return mc
        # some builds emit a top-level delta string
        if isinstance(obj.get("delta"), str) and obj.get("delta"):
            return obj.get("delta")
    except Exception:
        pass

    # Fallbacks
    for k in ("content", "token", "response"):
        v = obj.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _now_ts() -> float:
    return time.time()


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]

@functools.lru_cache(maxsize=16)
def _fetch_llamacpp_model_name(url: str) -> Optional[str]:
    try:
        r = requests.get(url.rstrip("/") + "/v1/models", timeout=2)
        if r.status_code == 200:
            data = r.json()
            if "data" in data and data["data"]:
                model_id = data["data"][0].get("id")
                return clean_model_name(model_id)
    except Exception:
        pass
    return None

def clean_model_name(raw: str) -> str:
    """Return a cleaned display name from a llama.cpp model id or path."""
    if not raw:
        return raw
    # 1. Keep only the filename
    name = Path(raw).name
    # 2. Remove extension
    name = re.sub(r"\.(gguf|bin|pt|onnx)$", "", name, flags=re.IGNORECASE)
    # 3. Truncate after first <number><b/B>
    m = re.search(r"\d+\s*[bB]", name)
    if m:
        name = name[: m.end()].rstrip("-_.")
    return name

def get_display_name(sd: Dict[str, Any]) -> str:
    side_letter = sd.get("name", "?")
    base = f"Side {side_letter}"
    mode = sd.get("mode")

    if mode == "Remote Server" and sd.get("remote_kind") == "llama.cpp":
        url = (sd.get("llamacpp_remote") or {}).get("url")
        model = _fetch_llamacpp_model_name(url) if url else None
        return model or "llama.cpp (remote)"

    if mode == "Remote Server" and sd.get("remote_kind") == "Ollama":
        return clean_model_name((sd.get("ollama") or {}).get("model")) or base

    if mode == "Host llama.cpp":
        mp = (sd.get("llamacpp_host") or {}).get("model_path")
        return clean_model_name(mp) if mp else base

    if mode == "ChatGPT API":
        return (sd.get("openai") or {}).get("model", base)

    return base


# ============================
# Sidebar — two independent sides
# ============================
models_map = list_local_gguf(MODELS_DIR)

with st.sidebar:
    st.header("Configuration")
    st.caption("Set up Side A and Side B, then run a side-by-side compare.")

    def side_config_ui(side_key: str, default_port: int):
        sd = st.session_state[side_key]
        st.subheader(get_display_name(sd))
        sd["mode"] = st.radio(
            f"Mode (Side {sd['name']})",
            ["Remote Server", "Host llama.cpp", "ChatGPT API"],
            index=["Remote Server", "Host llama.cpp", "ChatGPT API"].index(sd.get("mode", "Remote Server")),
            key=f"mode_{sd['name']}",
        )

        if sd["mode"] == "Remote Server":
            sd["remote_kind"] = st.radio(
                f"Remote kind (Side {sd['name']})",
                ["Ollama", "llama.cpp"],
                index=["Ollama", "llama.cpp"].index(sd.get("remote_kind", "Ollama")),
                key=f"rk_{sd['name']}",
            )

            if sd["remote_kind"] == "Ollama":
                ro = sd["ollama"]
                ro["url"] = st.text_input(f"Ollama URL (Side {sd['name']})", ro.get("url", DEFAULT_OLLAMA_URL), key=f"oll_url_{sd['name']}")

                c0, c1 = st.columns([2,1])
                opts = ro.get("models_from_server") or SUGGESTED_OLLAMA_MODELS
                with c0:
                    sel_idx = opts.index(ro.get("model", opts[0])) if ro.get("model") in opts else 0
                    ro["model"] = st.selectbox(f"Ollama model tag (Side {sd['name']})", options=opts, index=sel_idx, key=f"oll_model_{sd['name']}")
                with c1:
                    if st.button(f"↻ Refresh", key=f"oll_refresh_{sd['name']}"):
                        try:
                            resp = requests.get(ro["url"].rstrip("/") + "/api/tags", timeout=5)
                            models = []
                            for m in (resp.json().get("models") or []):
                                nm = m.get("name")
                                if nm:
                                    models.append(nm)
                            ro["models_from_server"] = sorted(set(models))
                            st.success(f"Side {sd['name']}: {len(ro['models_from_server'])} models found.")
                        except Exception as e:
                            st.error(f"Side {sd['name']}: could not fetch tags — {e}")

                ro["native_api"] = st.checkbox(
                    f"Use Ollama native API (/api/chat) (Side {sd['name']})",
                    value=bool(ro.get("native_api", True)),
                    key=f"oll_native_{sd['name']}"
                )

                st.caption("Sampling — Ollama (options)")
                s = ro["sampling"]
                a1, a2, a3 = st.columns(3)
                s["temperature"] = a1.number_input(f"Temperature ({sd['name']})", 0.0, 2.0, float(s.get("temperature", 0.0)), 0.05, key=f"oll_temp_{sd['name']}")
                s["top_p"] = a2.number_input(f"Top-p ({sd['name']})", 0.0, 1.0, float(s.get("top_p", 0.95)), 0.01, key=f"oll_topp_{sd['name']}")
                s["top_k"] = int(a3.number_input(f"Top-k ({sd['name']})", 0, 10000, int(s.get("top_k", 40)), 1, key=f"oll_topk_{sd['name']}") )
                b1, b2, b3 = st.columns(3)
                s["min_p"] = b1.number_input(f"Min-p ({sd['name']})", 0.0, 1.0, float(s.get("min_p", 0.05)), 0.01, key=f"oll_minp_{sd['name']}")
                s["typical_p"] = b2.number_input(f"Typical-p ({sd['name']})", 0.0, 1.0, float(s.get("typical_p", 1.0)), 0.01, key=f"oll_typ_{sd['name']}")
                s["repeat_penalty"] = b3.number_input(f"Repeat penalty ({sd['name']})", 0.0, 3.0, float(s.get("repeat_penalty", 1.0)), 0.01, key=f"oll_rep_{sd['name']}")
                c1, c2, c3 = st.columns(3)
                s["presence_penalty"] = c1.number_input(f"Presence penalty ({sd['name']})", -2.0, 2.0, float(s.get("presence_penalty", 0.0)), 0.01, key=f"oll_pres_{sd['name']}")
                s["frequency_penalty"] = c2.number_input(f"Frequency penalty ({sd['name']})", -2.0, 2.0, float(s.get("frequency_penalty", 0.0)), 0.01, key=f"oll_freq_{sd['name']}")
                s["tfs_z"] = c3.number_input(f"TFS-z ({sd['name']})", 0.0, 1.0, float(s.get("tfs_z", 1.0)), 0.01, key=f"oll_tfsz_{sd['name']}")
                d1, d2, d3 = st.columns(3)
                s["mirostat"] = d1.selectbox(f"Mirostat ({sd['name']})", [0, 1, 2], index=[0,1,2].index(int(s.get("mirostat", 0))), key=f"oll_miro_{sd['name']}")
                s["mirostat_tau"] = d2.number_input(f"Mirostat τ ({sd['name']})", 0.0, 10.0, float(s.get("mirostat_tau", 5.0)), 0.1, key=f"oll_tau_{sd['name']}")
                s["mirostat_eta"] = d3.number_input(f"Mirostat η ({sd['name']})", 0.0, 1.0, float(s.get("mirostat_eta", 0.1)), 0.01, key=f"oll_eta_{sd['name']}")
                e1, e2 = st.columns(2)
                s["seed"] = int(e1.number_input(f"Seed (0=auto) ({sd['name']})", 0, 2**31-1, int(s.get("seed", 0)), 1, key=f"oll_seed_{sd['name']}") )
                s["num_ctx"] = int(e2.number_input(f"Context tokens (num_ctx) ({sd['name']})", 4096, 262144, int(s.get("num_ctx", 8192)), 256, key=f"oll_ctx_{sd['name']}") )
                s["num_predict"] = int(st.number_input(f"Max output tokens (num_predict) ({sd['name']})", 0, 131072, int(s.get("num_predict", 4096)), 16, key=f"oll_predict_{sd['name']}") )

            else:  # llama.cpp remote
                rl = sd["llamacpp_remote"]
                rl["url"] = st.text_input(f"llama.cpp server URL (Side {sd['name']})", rl.get("url", DEFAULT_LLAMA_CPP_URL), key=f"lcr_url_{sd['name']}")
                st.caption("Sampling — llama.cpp (remote)")
                s = rl["sampling"]
                a1, a2, a3 = st.columns(3)
                s["temperature"] = a1.number_input(f"Temperature ({sd['name']})", 0.0, 2.0, float(s.get("temperature", 0.0)), 0.05, key=f"lcr_temp_{sd['name']}")
                s["top_p"] = a2.number_input(f"Top-p ({sd['name']})", 0.0, 1.0, float(s.get("top_p", 0.95)), 0.01, key=f"lcr_topp_{sd['name']}")
                s["top_k"] = int(a3.number_input(f"Top-k ({sd['name']})", 0, 10000, int(s.get("top_k", 40)), 1, key=f"lcr_topk_{sd['name']}") )
                b1, b2, b3 = st.columns(3)
                s["min_p"] = b1.number_input(f"Min-p ({sd['name']})", 0.0, 1.0, float(s.get("min_p", 0.05)), 0.01, key=f"lcr_minp_{sd['name']}")
                s["typical_p"] = b2.number_input(f"Typical-p ({sd['name']})", 0.0, 1.0, float(s.get("typical_p", 1.0)), 0.01, key=f"lcr_typ_{sd['name']}")
                s["repeat_penalty"] = b3.number_input(f"Repeat penalty ({sd['name']})", 0.0, 3.0, float(s.get("repeat_penalty", 1.0)), 0.01, key=f"lcr_rep_{sd['name']}")
                c1, c2, c3 = st.columns(3)
                s["presence_penalty"] = c1.number_input(f"Presence penalty ({sd['name']})", -2.0, 2.0, float(s.get("presence_penalty", 0.0)), 0.01, key=f"lcr_pres_{sd['name']}")
                s["frequency_penalty"] = c2.number_input(f"Frequency penalty ({sd['name']})", -2.0, 2.0, float(s.get("frequency_penalty", 0.0)), 0.01, key=f"lcr_freq_{sd['name']}")
                s["tfs_z"] = c3.number_input(f"TFS-z ({sd['name']})", 0.0, 1.0, float(s.get("tfs_z", 1.0)), 0.01, key=f"lcr_tfsz_{sd['name']}")
                d1, d2, d3 = st.columns(3)
                s["mirostat"] = d1.selectbox(f"Mirostat ({sd['name']})", [0, 1, 2], index=[0,1,2].index(int(s.get("mirostat", 0))), key=f"lcr_miro_{sd['name']}")
                s["mirostat_tau"] = d2.number_input(f"Mirostat τ ({sd['name']})", 0.0, 10.0, float(s.get("mirostat_tau", 5.0)), 0.1, key=f"lcr_tau_{sd['name']}")
                s["mirostat_eta"] = d3.number_input(f"Mirostat η ({sd['name']})", 0.0, 1.0, float(s.get("mirostat_eta", 0.1)), 0.01, key=f"lcr_eta_{sd['name']}")
                rl["max_tokens"] = int(st.number_input(f"Max output tokens ({sd['name']})", 0, 131072, int(rl.get("max_tokens", 4096)), 16, key=f"lcr_max_{sd['name']}") )

        elif sd["mode"] == "Host llama.cpp":
            hl = sd["llamacpp_host"]
            model_names = list(models_map.keys())
            if not model_names:
                st.error("No GGUF models found in ./models")
            else:
                sel = st.selectbox(f"Model (GGUF) for Side {sd['name']}", options=model_names, index=0, key=f"host_model_{sd['name']}")
                hl["model_path"] = str(models_map[sel])
            hl["port"] = int(st.number_input(f"Server port (Side {sd['name']})", 1, 65535, int(hl.get("port", default_port)), key=f"host_port_{sd['name']}") )
            hl["threads"] = int(st.number_input(f"Threads (Side {sd['name']})", 1, 256, int(hl.get("threads", DEFAULT_THREADS)), key=f"host_thr_{sd['name']}") )
            c1, c2 = st.columns(2)
            if c1.button(f"🚀 Launch / Restart (Side {sd['name']})", key=f"host_launch_{sd['name']}"):
                try:
                    if not hl.get("model_path"):
                        st.error("Please choose a model.")
                    else:
                        start_host_llamacpp(side_key, Path(hl["model_path"]), hl["port"], hl["threads"])
                        st.success(f"Side {sd['name']}: llama.cpp on {CLIENT_HOST}:{hl['port']}")
                except ServerLaunchError as e:
                    st.error(str(e))
            if c2.button(f"🛑 Stop (Side {sd['name']})", key=f"host_stop_{sd['name']}"):
                stop_host_llamacpp(side_key); st.info(f"Side {sd['name']}: server stopped.")

            st.caption("Sampling — llama.cpp (hosted)")
            # Reuse remote llama.cpp sampling UI for hosted
            rl_tmp = sd["llamacpp_remote"]
            s = rl_tmp["sampling"]
            a1, a2, a3 = st.columns(3)
            s["temperature"] = a1.number_input(f"Temperature ({sd['name']})", 0.0, 2.0, float(s.get("temperature", 0.0)), 0.05, key=f"hl_temp_{sd['name']}")
            s["top_p"] = a2.number_input(f"Top-p ({sd['name']})", 0.0, 1.0, float(s.get("top_p", 0.95)), 0.01, key=f"hl_topp_{sd['name']}")
            s["top_k"] = int(a3.number_input(f"Top-k ({sd['name']})", 0, 10000, int(s.get("top_k", 40)), 1, key=f"hl_topk_{sd['name']}") )
            b1, b2, b3 = st.columns(3)
            s["min_p"] = b1.number_input(f"Min-p ({sd['name']})", 0.0, 1.0, float(s.get("min_p", 0.05)), 0.01, key=f"hl_minp_{sd['name']}")
            s["typical_p"] = b2.number_input(f"Typical-p ({sd['name']})", 0.0, 1.0, float(s.get("typical_p", 1.0)), 0.01, key=f"hl_typ_{sd['name']}")
            s["repeat_penalty"] = b3.number_input(f"Repeat penalty ({sd['name']})", 0.0, 3.0, float(s.get("repeat_penalty", 1.0)), 0.01, key=f"hl_rep_{sd['name']}")
            c1, c2, c3 = st.columns(3)
            s["presence_penalty"] = c1.number_input(f"Presence penalty ({sd['name']})", -2.0, 2.0, float(s.get("presence_penalty", 0.0)), 0.01, key=f"hl_pres_{sd['name']}")
            s["frequency_penalty"] = c2.number_input(f"Frequency penalty ({sd['name']})", -2.0, 2.0, float(s.get("frequency_penalty", 0.0)), 0.01, key=f"hl_freq_{sd['name']}")
            s["tfs_z"] = c3.number_input(f"TFS-z ({sd['name']})", 0.0, 1.0, float(s.get("tfs_z", 1.0)), 0.01, key=f"hl_tfsz_{sd['name']}")
            d1, d2, d3 = st.columns(3)
            s["mirostat"] = d1.selectbox(f"Mirostat ({sd['name']})", [0, 1, 2], index=[0,1,2].index(int(s.get("mirostat", 0))), key=f"hl_miro_{sd['name']}")
            s["mirostat_tau"] = d2.number_input(f"Mirostat τ ({sd['name']})", 0.0, 10.0, float(s.get("mirostat_tau", 5.0)), 0.1, key=f"hl_tau_{sd['name']}")
            s["mirostat_eta"] = d3.number_input(f"Mirostat η ({sd['name']})", 0.0, 1.0, float(s.get("mirostat_eta", 0.1)), 0.01, key=f"hl_eta_{sd['name']}")
            rl_tmp["max_tokens"] = int(st.number_input(f"Max output tokens ({sd['name']})", 0, 131072, int(rl_tmp.get("max_tokens", 4096)), 16, key=f"hl_max_{sd['name']}") )

        else:  # ChatGPT API
            oa = sd["openai"]
            if oa.get("api_key"):
                st.success(f"Side {sd['name']}: Using OPENAI_API_KEY from env/previous input")
            oa["api_key"] = st.text_input(
                f"OpenAI API Key (Side {sd['name']})",
                value=(oa.get("api_key") or ""),
                type="password",
                key=f"oa_key_{sd['name']}"
            ) or oa.get("api_key")

            st.markdown("*Pick a model from the supported list (per OpenAI docs).*")
            more = st.checkbox(f"Show more models (Side {sd['name']})", value=False, key=f"oa_more_{sd['name']}")
            model_list = OPENAI_CORE_MODELS + (OPENAI_MORE_MODELS if more else [])
            cur = oa.get("model", OPENAI_CORE_MODELS[0])
            sel_idx = model_list.index(cur) if cur in model_list else 0
            oa["model"] = st.selectbox(f"OpenAI model (Side {sd['name']})", model_list, index=sel_idx, key=f"oa_model_{sd['name']}")

            # Sampling knobs
            s = oa["sampling"]
            a1, a2 = st.columns(2)
            s["temperature"] = a1.number_input(f"Temperature ({sd['name']})", 0.0, 2.0, float(s.get("temperature", 0.0)), 0.05, key=f"oa_temp_{sd['name']}")
            s["top_p"] = a2.number_input(f"Top-p ({sd['name']})", 0.0, 1.0, float(s.get("top_p", 0.95)), 0.01, key=f"oa_topp_{sd['name']}")
            b1, b2 = st.columns(2)
            s["presence_penalty"] = b1.number_input(f"Presence penalty ({sd['name']})", -2.0, 2.0, float(s.get("presence_penalty", 0.0)), 0.01, key=f"oa_pres_{sd['name']}")
            s["frequency_penalty"] = b2.number_input(f"Frequency penalty ({sd['name']})", -2.0, 2.0, float(s.get("frequency_penalty", 0.0)), 0.01, key=f"oa_freq_{sd['name']}")

            st.caption("Note: some models (e.g., GPT-5 & o3 families) reject sampling knobs like temperature/top_p; the app auto-retries without them. Presence/frequency penalties are not supported by the Responses API.")

            c1, c2 = st.columns(2)
            oa["verbosity"] = c1.selectbox(f"Verbosity ({sd['name']})", ["low", "medium", "high"], index=["low","medium","high"].index(oa.get("verbosity", "medium")), key=f"oa_verb_{sd['name']}")
            oa["reasoning_effort"] = c2.selectbox(f"Reasoning effort ({sd['name']})", ["minimal", "low", "medium", "high"], index=["minimal","low","medium","high"].index(oa.get("reasoning_effort", "medium")), key=f"oa_eff_{sd['name']}")
            oa["max_output_tokens"] = int(st.number_input(f"Max output tokens ({sd['name']})", 1, 131072, int(oa.get("max_output_tokens", 4096)), 16, key=f"oa_max_{sd['name']}") )

            st.markdown("---")
            st.subheader(f"Tools (Side {sd['name']})")

            # Ensure dicts exist
            oa.setdefault("tools", {"web_search": True, "file_search": False, "code_interpreter": False})
            oa.setdefault("file_search", {"vector_store_ids": []})

            # Tool toggles
            tcol1, tcol2, tcol3 = st.columns(3)
            oa["tools"]["web_search"] = tcol1.checkbox("Web Search", value=bool(oa["tools"].get("web_search", True)), key=f"oa_tool_ws_{sd['name']}")
            oa["tools"]["file_search"] = tcol2.checkbox("File Search", value=bool(oa["tools"].get("file_search", False)), key=f"oa_tool_fs_{sd['name']}")
            oa["tools"]["code_interpreter"] = tcol3.checkbox("Code Interpreter", value=bool(oa["tools"].get("code_interpreter", False)), key=f"oa_tool_ci_{sd['name']}")

            # File Search settings (optional)
            if oa["tools"]["file_search"]:
                vs_raw = st.text_input(
                    f"File Search: Vector Store IDs (comma-separated) — Side {sd['name']}",
                    value=",".join(oa.get("file_search", {}).get("vector_store_ids", [])),
                    key=f"oa_fs_vs_{sd['name']}",
                    help="Provide one or more vector_store_ids you created in OpenAI."
                ).strip()
                oa["file_search"]["vector_store_ids"] = [x.strip() for x in vs_raw.split(",") if x.strip()]

            # Tool choice
            choice_map = {
                "Let model decide (auto)": "auto",
                "Disable tools (none)": "none",
                "Force Web Search": "web_search",
                "Force File Search": "file_search",
                "Force Code Interpreter": "code_interpreter",
            }
            inv_choice_map = {v: k for k, v in choice_map.items()}
            sel = st.selectbox(
                f"Tool choice ({sd['name']})",
                list(choice_map.keys()),
                index=list(choice_map.keys()).index(inv_choice_map.get(oa.get("tool_choice", "auto"), "Let model decide (auto)")),
                key=f"oa_tool_choice_{sd['name']}"
            )
            oa["tool_choice"] = choice_map[sel]

    with st.expander("Side A settings", expanded=True):
        side_config_ui("sideA", DEFAULT_PORT_A)
    with st.expander("Side B settings", expanded=True):
        side_config_ui("sideB", DEFAULT_PORT_B)

# ============================
# Prompt & Compare
# ============================
system_prompt = st.text_area("System prompt", value="You are a helpful assistant.")
prompt = st.text_area("Your message", value="Explain the difference between CPU and GPU in two paragraphs.")
col_go, col_clear = st.columns(2)
start = col_go.button("▶️ Start comparison", type="primary")
reset = col_clear.button("🧹 Clear")
if reset:
    st.session_state.last_output = {"A": "", "B": ""}
    st.session_state.last_thinking = {"A": None, "B": None}
    st.session_state.last_timeline = ""
    st.rerun()

# Toggle merged timeline visibility
show_timeline = st.checkbox("Show merged timeline", value=True, key="show_timeline")

# ============================
# Workers
# ============================
def _persist_thinking(which: str, st_state: Dict[str, Any]):
    st.session_state.last_thinking[which] = {
        "header": st_state.get("header", ""),
        "content": st_state.get("content", ""),
        "has_thinking": bool(st_state.get("detected_tags")),
    }
def _render_stored_thinking(which: str):
    data = (st.session_state.get("last_thinking") or {}).get(which)
    if not data or not data.get("has_thinking"):
        return
    placeholder = think_a if which == "A" else think_b
    title = f"🧠 Thinking — {data.get('header','')}" if data.get('header') else "🧠 Thinking…"
    with placeholder.container():
        with st.expander(title, expanded=False):
            st.markdown(data.get("content",""))
def _messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt.strip()})
    if user_prompt.strip():
        msgs.append({"role": "user", "content": user_prompt.strip()})
    return msgs
def worker_openai(name: str, cfg: Dict[str, Any], messages: List[Dict[str, str]], out_q, done_evt, err_q):
    try:
        from openai import OpenAI
    except Exception as e:
        err_q.put((name, f"OpenAI SDK not installed: {e}"))
        done_evt.set(); return

    oa = cfg["openai"]
    api_key = oa.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        err_q.put((name, "Missing OpenAI API key.")); done_evt.set(); return

    try:
        client = OpenAI(api_key=api_key)
        def to_responses_input(msgs: List[Dict[str, str]]):
            return [{"role": m.get("role", "user"), "content": [{"type": "input_text", "text": m.get("content", "")}]} for m in msgs]

        # --- Build request params (with Tools support) ---
        params: Dict[str, Any] = dict(
            model=oa.get("model", "gpt-5"),
            input=to_responses_input(messages),
            max_output_tokens=int(oa.get("max_output_tokens", 4096)),
            stream=True,
        )

        # Assemble tools from UI
        tools_cfg = (oa.get("tools") or {})
        tools_list = []

        if tools_cfg.get("web_search"):
            tools_list.append({"type": "web_search"})

        if tools_cfg.get("file_search"):
            fs_cfg = oa.get("file_search") or {}
            vs_ids = fs_cfg.get("vector_store_ids") or []
            if vs_ids:
                tools_list.append({"type": "file_search", "vector_store_ids": vs_ids})

        if tools_cfg.get("code_interpreter"):
            tools_list.append({"type": "code_interpreter"})

        if tools_list:
            params["tools"] = tools_list

        # tool_choice: "auto" | "none" | {"type": "..."}
        tc = str(oa.get("tool_choice", "auto"))
        if tc in ("auto", "none"):
            params["tool_choice"] = tc
        elif tc in ("web_search", "file_search", "code_interpreter"):
            params["tool_choice"] = {"type": tc}

        # Sampling knobs (fallback-safe)
        s = oa.get("sampling", {}) or {}
        if "temperature" in s:
            try:
                params["temperature"] = float(s.get("temperature", 0.0))
            except Exception:
                pass
        if "top_p" in s:
            try:
                params["top_p"] = float(s.get("top_p", 0.95))
            except Exception:
                pass

        # Extra controls
        params["text"] = {"verbosity": str(oa.get("verbosity", "medium"))}
        params["reasoning"] = {"effort": str(oa.get("reasoning_effort", "medium"))}

        # Create stream with graceful fallback for unsupported parameters
        def _create_with_fallback(p0: Dict[str, Any]):
            try:
                return client.responses.create(**p0)
            except Exception:
                # Remove sampling knobs and retry
                p1 = dict(p0)
                p1.pop("temperature", None)
                p1.pop("top_p", None)
                try:
                    return client.responses.create(**p1)
                except Exception:
                    # Remove extra controls and retry once more
                    p2 = dict(p1)
                    p2.pop("text", None)
                    p2.pop("reasoning", None)
                    # Keep tools; if a model rejects tools, OpenAI will return a clear error
                    return client.responses.create(**p2)

        try:
            stream = _create_with_fallback(params)
        except Exception as e:
            err_q.put((name, str(e))); done_evt.set(); return

        for event in stream:
            try:
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        out_q.put((delta, _now_ts()))
                elif getattr(event, "type", "") == "response.error":
                    err_obj = getattr(event, "error", None)
                    msg = getattr(err_obj, "message", None) or str(event)
                    err_q.put((name, msg))
                elif getattr(event, "type", "") == "response.completed":
                    break
            except Exception:
                pass
    except Exception as e:
        err_q.put((name, str(e)))
    finally:
        done_evt.set()
def worker_llamacpp_remote(name: str, cfg: Dict[str, Any], messages: List[Dict[str, str]], out_q, done_evt, err_q, url_override: Optional[str] = None, max_tokens: Optional[int] = None):
    rl = cfg["llamacpp_remote"]
    url_base = (url_override or rl.get("url", DEFAULT_LLAMA_CPP_URL)).rstrip("/")
    url = url_base + "/v1/chat/completions"
    s = rl.get("sampling", {})
    body = {
        "model": s.get("model", "llama"),  # ignored by server
        "messages": messages,
        "stream": True,
        "max_tokens": int(max_tokens if max_tokens is not None else rl.get("max_tokens", 4096)),
        "temperature": float(s.get("temperature", 0.0)),
        "top_p": float(s.get("top_p", 0.95)),
        "top_k": int(s.get("top_k", 40)),
        "min_p": float(s.get("min_p", 0.05)),
        "typical_p": float(s.get("typical_p", 1.0)),
        "repeat_penalty": float(s.get("repeat_penalty", 1.0)),
        "presence_penalty": float(s.get("presence_penalty", 0.0)),
        "frequency_penalty": float(s.get("frequency_penalty", 0.0)),
    }
    seed = int(s.get("seed", 0))
    if seed > 0:
        body["seed"] = seed

    try:
        with requests.post(url, headers=_headers_json(None), data=json.dumps(body), stream=True, timeout=120) as r:
            if r.status_code >= 400:
                raise StreamError(f"HTTP {r.status_code} — {r.text[:400]}")
            for obj in parse_sse_lines(r.iter_lines()):
                chunk = extract_delta_text(obj)
                if chunk:
                    out_q.put((chunk, _now_ts()))
    except Exception as e:
        err_q.put((name, str(e)))
    finally:
        done_evt.set()
def worker_ollama_remote(name: str, cfg: Dict[str, Any], messages: List[Dict[str, str]], out_q, done_evt, err_q):
    ro = cfg["ollama"]
    url_base = ro.get("url", DEFAULT_OLLAMA_URL).rstrip("/")
    model = ro.get("model", "llama3:8b")
    native = bool(ro.get("native_api", True))
    s = ro.get("sampling", {})

    if native:
        url = url_base + "/api/chat"
        options = {
            "temperature": float(s.get("temperature", 0.0)),
            "top_p": float(s.get("top_p", 0.95)),
            "top_k": int(s.get("top_k", 40)),
            "min_p": float(s.get("min_p", 0.05)),
            "typical_p": float(s.get("typical_p", 1.0)),
            "repeat_penalty": float(s.get("repeat_penalty", 1.0)),
            "presence_penalty": float(s.get("presence_penalty", 0.0)),
            "frequency_penalty": float(s.get("frequency_penalty", 0.0)),
            "num_ctx": int(s.get("num_ctx", 8192)),
            "num_predict": int(s.get("num_predict", 4096)),
        }
        seed = int(s.get("seed", 0))
        if seed > 0:
            options["seed"] = seed
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
            "keep_alive": str(ro.get("keep_alive", "5m")),
        }
    else:
        url = url_base + "/v1/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": float(s.get("temperature", 0.0)),
            "top_p": float(s.get("top_p", 0.95)),
            "presence_penalty": float(s.get("presence_penalty", 0.0)),
            "frequency_penalty": float(s.get("frequency_penalty", 0.0)),
            "max_tokens": int(s.get("num_predict", 4096)),
        }

    try:
        with requests.post(url, headers=_headers_json(None), data=json.dumps(body), stream=True, timeout=180) as r:
            if r.status_code >= 400:
                raise StreamError(f"HTTP {r.status_code} — {r.text[:400]}")
            for obj in parse_sse_lines(r.iter_lines()):
                chunk = extract_delta_text(obj)
                if chunk:
                    out_q.put((chunk, _now_ts()))
    except Exception as e:
        err_q.put((name, str(e)))
    finally:
        done_evt.set()

# ============================
# Kick off comparison
# ============================

# Helper to decorate caption with active tools (for ChatGPT side only)
def _active_tools_badge(sd: Dict[str, Any]) -> str:
    if sd.get("mode") != "ChatGPT API":
        return ""
    oa = sd.get("openai", {})
    t = oa.get("tools", {}) or {}
    active = [k.replace("_", " ").title() for k, v in t.items() if v]
    return (" · Tools: " + ", ".join(active)) if active else " · Tools: none"

# --- Always-on output boxes (so persisted content can show if you keep it elsewhere) ---
colA, colB = st.columns(2)
with colA:
    srcA = st.session_state.sideA
    st.subheader(f"{get_display_name(srcA)}")
    a_desc = srcA["mode"] + (f" → {srcA['remote_kind']}" if srcA["mode"] == "Remote Server" else "") + _active_tools_badge(srcA)
    st.caption(a_desc)
    think_a = st.empty()   # thinking section placeholder (GPT-OSS)
    box_a = st.empty(); meta_a = st.empty()

with colB:
    srcB = st.session_state.sideB
    st.subheader(f"{get_display_name(srcB)}")
    b_desc = srcB["mode"] + (f" → {srcB['remote_kind']}" if srcB["mode"] == "Remote Server" else "") + _active_tools_badge(srcB)
    st.caption(b_desc)
    think_b = st.empty()   # thinking section placeholder (GPT-OSS)
    box_b = st.empty(); meta_b = st.empty()

# Merged timeline placeholder
if show_timeline:
    st.markdown("---")
    st.subheader("Merged timeline")
    timeline_box = st.empty()
else:
    timeline_box = None

# (Optional) re-render persisted content if you maintain it in session_state
box_a.markdown(st.session_state.get("last_output", {}).get("A", "") if "last_output" in st.session_state else "")
box_b.markdown(st.session_state.get("last_output", {}).get("B", "") if "last_output" in st.session_state else "")
if show_timeline and "last_timeline" in st.session_state and st.session_state.last_timeline:
    timeline_box.markdown(st.session_state.last_timeline)
_render_stored_thinking("A")
_render_stored_thinking("B")
# --------- GPT-OSS tag-aware streaming helpers (live expander) ----------
_GPTOSS_START = "<|channel|>"
_GPTOSS_MSG   = "<|message|>"
_GPTOSS_END   = "<|end|>"

import re
_tag_re = re.compile(r"<\|.*?\|>", re.DOTALL)

def _clean_brace_text(s: str) -> str:
    # Remove <|...|> tags, outer braces, and trim
    s = _tag_re.sub("", s).strip()
    if s.startswith("{"):
        s = s[1:]
    if s.endswith("}"):
        s = s[:-1]
    return s.strip()

def _init_side_stream_state():
    return {
        "raw": "",
        "detected_tags": False,      # saw <|channel|>
        "thinking_done": False,      # saw <|end|>
        "end_pos": -1,

        # live thinking (partial supported)
        "header": "",
        "content": "",

        # expander mount/update
        "expander_mounted": False,
        "expander_title": "🧠 Thinking…",
        "content_ph": None,          # st.empty() inside the expander

        # answer streaming after <|end|>
        "visible_len": 0,
    }

def _mount_or_update_thinking(which: str, s: Dict[str, Any]):
    """Mount the expander ASAP and update header/content during streaming."""
    if not s["detected_tags"]:
        return
    placeholder = think_a if which == "A" else think_b

    # first mount
    if not s["expander_mounted"]:
        placeholder.empty()
        with placeholder.container():
            with st.expander(s.get("expander_title", "🧠 Thinking…"), expanded=False):
                s["content_ph"] = st.empty()
                s["content_ph"].markdown(s["content"])
        s["expander_mounted"] = True
        return

    # title may change once header is available -> remount to update title
    new_title = f"🧠 Thinking — {s['header']}" if s["header"] else "🧠 Thinking…"
    if new_title != s["expander_title"]:
        s["expander_title"] = new_title
        placeholder.empty()
        with placeholder.container():
            with st.expander(s["expander_title"], expanded=False):
                s["content_ph"] = st.empty()
                s["content_ph"].markdown(s["content"])
    else:
        if s["content_ph"] is not None:
            s["content_ph"].markdown(s["content"])

def _process_chunk(side_state: Dict[str, Any], chunk: str) -> str:
    """
    Returns the delta for the visible 'answer'.
    - If GPT-OSS tags are detected, we show 'thinking' live in the expander
      and suppress visible output until <|end|>. Then we stream only the text
      after <|end|>.
    - If no tags, stream chunks as-is.
    """
    s = side_state
    s["raw"] += chunk

    # already after end -> stream answer part
    if s["thinking_done"]:
        visible = s["raw"][s["end_pos"] + len(_GPTOSS_END):]
        delta = visible[s["visible_len"]:]
        s["visible_len"] = len(visible)
        return delta

    # detect tag presence
    if not s["detected_tags"] and (_GPTOSS_START in s["raw"]):
        s["detected_tags"] = True

    if s["detected_tags"]:
        raw = s["raw"]
        i = raw.find(_GPTOSS_START)
        j = raw.find(_GPTOSS_MSG, i + len(_GPTOSS_START)) if i != -1 else -1
        k = raw.find(_GPTOSS_END, j + len(_GPTOSS_MSG)) if j != -1 else -1

        # progressively extract header
        if i != -1:
            hdr_seg = raw[i + len(_GPTOSS_START): (j if j != -1 else len(raw))]
            s["header"] = _clean_brace_text(hdr_seg)

        # progressively extract content
        if j != -1:
            cnt_seg = raw[j + len(_GPTOSS_MSG): (k if k != -1 else len(raw))]
            s["content"] = _clean_brace_text(cnt_seg)

        # reached end marker -> switch to answer streaming
        if i != -1 and j != -1 and k != -1:
            s["thinking_done"] = True
            s["end_pos"] = k
            visible = raw[k + len(_GPTOSS_END):]
            s["visible_len"] = len(visible)
            return visible

        # until <|end|>, do not show any answer text
        return ""

    # no tags -> normal streaming
    return chunk

# ============================
# Start comparison
# ============================
if start:
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    messages = _messages(system_prompt, prompt)

    import threading, queue
    q_a, q_b = queue.Queue(), queue.Queue()
    done_a, done_b = threading.Event(), threading.Event()
    err_q = queue.Queue()

    def _snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
        s = copy.deepcopy(cfg)
        # Drop non-serializable runtime handles (e.g., subprocess.Popen)
        try:
            if "llamacpp_host" in s and "proc" in s["llamacpp_host"]:
                s["llamacpp_host"]["proc"] = None
        except Exception:
            pass
        return s

    snapA = _snapshot(st.session_state.sideA)
    snapB = _snapshot(st.session_state.sideB)

    # Resolve workers per side
    def launch_worker(label: str, cfg: Dict[str, Any], out_q, done_evt):
        mode = cfg.get("mode")
        if mode == "ChatGPT API":
            return threading.Thread(target=worker_openai, args=(label, cfg, messages, out_q, done_evt, err_q), daemon=True)
        elif mode == "Remote Server":
            if cfg.get("remote_kind") == "Ollama":
                return threading.Thread(target=worker_ollama_remote, args=(label, cfg, messages, out_q, done_evt, err_q), daemon=True)
            else:
                return threading.Thread(target=worker_llamacpp_remote, args=(label, cfg, messages, out_q, done_evt, err_q), daemon=True)
        elif mode == "Host llama.cpp":
            hl = cfg.get("llamacpp_host", {})
            port = int(hl.get("port", DEFAULT_PORT_A))
            base = f"http://{CLIENT_HOST}:{port}"
            # Health check instead of inspecting a Popen handle
            ok = False
            for path in ("/health", "/healthz", "/"):
                try:
                    r = requests.get(base + path, timeout=1)
                    if r.status_code < 500:
                        ok = True
                        break
                except Exception:
                    pass
            if not ok:
                st.error(f"Side {label[-1]}: llama.cpp not reachable at {base}. Launch it in the sidebar.")
                return None
            url = base
            return threading.Thread(
                target=worker_llamacpp_remote,
                args=(label, cfg, messages, out_q, done_evt, err_q, url, cfg.get("llamacpp_remote", {}).get("max_tokens", 4096)),
                daemon=True,
            )
        else:
            st.error(f"Side {label[-1]}: Unknown mode {mode}")
            return None

    t_a = launch_worker("A (" + st.session_state.sideA["mode"] + ")", snapA, q_a, done_a)
    t_b = launch_worker("B (" + st.session_state.sideB["mode"] + ")", snapB, q_b, done_b)

    if t_a is None or t_b is None:
        st.stop()

    t_a.start(); t_b.start()

    # Fresh buffers for this run
    buf_a, buf_b, timeline = [], [], []
    last_render = 0.0

    # Per-side GPT-OSS parsing state
    state_a = _init_side_stream_state()
    state_b = _init_side_stream_state()

    while True:
        try:
            name_e, err = err_q.get_nowait()
            st.error(f"{name_e} error: {err}")
        except queue.Empty:
            pass

        progressed = False

        # A-side
        try:
            chunk, ts = q_a.get_nowait()
            delta = _process_chunk(state_a, chunk)
            _mount_or_update_thinking("A", state_a)  # live update expander
            if delta:
                buf_a.append(delta)
                timeline.append((ts, get_display_name(snapA), delta))
                timeline.append((ts, "A", delta))
                progressed = True
        except queue.Empty:
            pass

        # B-side
        try:
            chunk, ts = q_b.get_nowait()
            delta = _process_chunk(state_b, chunk)
            _mount_or_update_thinking("B", state_b)  # live update expander
            if delta:
                buf_b.append(delta)
                timeline.append((ts, get_display_name(snapB), delta))
                timeline.append((ts, "B", delta))
                progressed = True
        except queue.Empty:
            pass

        now = _now_ts()
        if progressed or (now - last_render) > 0.05:
            box_a.markdown("".join(buf_a))
            box_b.markdown("".join(buf_b))
            if buf_a:
                last_a = max([t for t,n,_ in timeline if n=='A'], default=_now_ts())
                meta_a.caption(f"Last chunk: {_fmt_ts(last_a)}")
            if buf_b:
                last_b = max([t for t,n,_ in timeline if n=='B'], default=_now_ts())
                meta_b.caption(f"Last chunk: {_fmt_ts(last_b)}")
            if show_timeline and timeline_box and timeline:
                tl = sorted(timeline, key=lambda x: x[0])[-200:]
                lines = [f"`{_fmt_ts(ts)}` **{name}**: {text}" for ts, name, text in tl]
                timeline_box.markdown("".join(lines))
            last_render = now

        if done_a.is_set() and done_b.is_set() and q_a.empty() and q_b.empty():
            break
        time.sleep(0.01)
    _persist_thinking("A", state_a)
    _persist_thinking("B", state_b)
    st.success("Both streams finished.")

    # (Optional) persist final outputs if you keep them
    if "last_output" in st.session_state:
        st.session_state.last_output["A"] = "".join(buf_a)
        st.session_state.last_output["B"] = "".join(buf_b)
    if show_timeline and timeline and "last_timeline" in st.session_state:
        tl = sorted(timeline, key=lambda x: x[0])[-200:]
        st.session_state.last_timeline = "".join(
            f"`{_fmt_ts(ts)}` **{name}**: {text}" for ts, name, text in tl
        )

st.info(
    "Each side is fully independent: choose Remote (Ollama or llama.cpp), Host a local llama.cpp with a GGUF from ./models, or ChatGPT API. "
    "On the ChatGPT API side you can enable tools like Web Search, File Search (needs vector_store_ids), and Code Interpreter."
)
