import os
import sys
import re
import json
import time
import shutil
import socket
import hashlib
import importlib
import subprocess
import threading
import contextlib
import http.server
import socketserver
from http import HTTPStatus
from urllib.parse import urlparse, parse_qs, unquote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

term_width = shutil.get_terminal_size().columns
LEFT_WIDTH = int(term_width * 0.55)

VERSION = "1.0.0"
AUTO_UPDATE_LLAMA = True  
HTTP_PORT = 8080  
_AUTO_UPDATE_FLAG = "_AUTO_UPDATE_IN_PROGRESS"  

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

HAS_AI_ENGINE = False
LAST_UPDATE_STATUS = None
http_server = None
loader = None
_LLAMA_LOG_CONFIGURED = False
_LLAMA_LOG_CALLBACK = None
TERMINAL_CHAT_HISTORY: List[Dict[str, str]] = []

RECOMMENDED_RUNTIME_PACKAGES = [
    "llama-cpp-python",
]

KNOWN_UNSUPPORTED_ARCH_BY_MAX_VERSION: Dict[str, Tuple[int, int, int]] = {}

def _is_env_flag_enabled(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    value = os.environ.get(name, fallback).strip().lower()
    return value in {"1", "true", "yes", "on"}

def load_model_aliases():
    aliases = {}
    
    if not os.path.exists("models"):
        return aliases
    
    for filename in os.listdir("models"):
        if filename.endswith('.gguf') and 'mmproj' not in filename.lower():
            alias_name = filename[:-5]
            aliases[alias_name] = {
                "url": f"file://{filename}",
                "filename": filename
            }
    
    return aliases

def _parse_version_tuple(version_text: str) -> Tuple[int, int, int]:
    parts = (version_text or "").strip().split(".")
    out: List[int] = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)  

def _is_arch_known_unsupported(arch: str, llama_version: str) -> bool:
    max_version = KNOWN_UNSUPPORTED_ARCH_BY_MAX_VERSION.get((arch or "").strip().lower())
    if not max_version:
        return False
    return _parse_version_tuple(llama_version) <= max_version

def _read_gguf_architecture(model_path: str) -> Optional[str]:
    def _read_exact(handle, size: int) -> bytes:
        data = handle.read(size)
        if len(data) != size:
            raise EOFError
        return data

    def _read_u32(handle) -> int:
        return int.from_bytes(_read_exact(handle, 4), "little", signed=False)

    def _read_u64(handle) -> int:
        return int.from_bytes(_read_exact(handle, 8), "little", signed=False)

    def _read_str(handle) -> str:
        size = _read_u64(handle)
        return _read_exact(handle, size).decode("utf-8", errors="ignore")

    def _skip_value(handle, value_type: int) -> None:
        if value_type in (0, 1, 7):
            _read_exact(handle, 1)
        elif value_type in (2, 3):
            _read_exact(handle, 2)
        elif value_type in (4, 5, 6):
            _read_exact(handle, 4)
        elif value_type in (10, 11, 12):
            _read_exact(handle, 8)
        elif value_type == 8:
            _read_exact(handle, _read_u64(handle))
        elif value_type == 9:
            element_type = _read_u32(handle)
            count = _read_u64(handle)
            for _ in range(count):
                _skip_value(handle, element_type)
        else:
            raise ValueError(f"Unknown GGUF value type: {value_type}")

    try:
        with open(model_path, "rb") as handle:
            if _read_exact(handle, 4) != b"GGUF":
                return None
            version = _read_u32(handle)
            if version < 2:
                return None
            _ = _read_u64(handle)
            kv_count = _read_u64(handle)
            for _ in range(kv_count):
                key = _read_str(handle)
                value_type = _read_u32(handle)
                if key == "general.architecture" and value_type == 8:
                    return _read_str(handle).strip().lower()
                _skip_value(handle, value_type)
    except Exception:
        return None
    return None

def _configure_llama_logging(llama_cpp) -> None:
    global _LLAMA_LOG_CONFIGURED, _LLAMA_LOG_CALLBACK
    if _LLAMA_LOG_CONFIGURED:
        return
    _LLAMA_LOG_CONFIGURED = True

    if os.environ.get("RUN_AI_VERBOSE_LLAMA", "").strip().lower() in {"1", "true", "yes"}:
        return

    try:
        @llama_cpp.llama_log_callback
        def _quiet_log(level, text, user_data):
            return

        _LLAMA_LOG_CALLBACK = _quiet_log
        llama_cpp.llama_log_set(_LLAMA_LOG_CALLBACK, None)
    except Exception:
        pass

def _patch_llama_model_del(llama_cpp) -> None:
    try:
        internals = getattr(llama_cpp, "_internals", None)
        model_cls = getattr(internals, "LlamaModel", None) if internals else None
        if model_cls is None or getattr(model_cls, "_run_ai_safe_del_patched", False):
            return

        original_del = getattr(model_cls, "__del__", None)
        if not callable(original_del):
            return

        def _safe_del(self):
            try:
                original_del(self)
            except AttributeError:
                pass
            except Exception:
                pass

        model_cls.__del__ = _safe_del
        model_cls._run_ai_safe_del_patched = True
    except Exception:
        pass

def ensure_llama_cpp():
    try:
        import llama_cpp
        _configure_llama_logging(llama_cpp)
        _patch_llama_model_del(llama_cpp)
        return llama_cpp
    except Exception:
        return None

class OutputFilter:
    def __init__(self):
        self.buffer = ""
        self.in_tag = False
        self.current_tag = ""
        self.keep_tags = ["/im_end", "im_end"]
    
    def feed(self, text: str) -> str:
        self.buffer += text
        result = []
        i = 0
        n = len(self.buffer)
        
        while i < n:
            if self.buffer[i] == '<' and i + 1 < n and self.buffer[i+1] == '|':
                tag_start = i
                i += 2  
                tag_name = ""
                
                while i < n and self.buffer[i] != '|' and self.buffer[i] != '>':
                    tag_name += self.buffer[i]
                    i += 1
                
                if i < n and self.buffer[i] == '|' and i + 1 < n and self.buffer[i+1] == '>':
                    i += 2  
                    tag = f"<|{tag_name}|>"
                    
                    if tag_name.startswith("im_start") or tag_name.startswith("system") or tag_name.startswith("user"):
                        self.in_tag = True
                        self.current_tag = tag_name
                        continue
                    elif tag_name.startswith("/im_start") or tag_name.startswith("/system") or tag_name.startswith("/user"):
                        self.in_tag = False
                        self.current_tag = ""
                        continue
                    elif tag_name in self.keep_tags:
                        result.append(tag)
                else:
                    result.append(self.buffer[tag_start:i])
            else:
                if not self.in_tag:
                    result.append(self.buffer[i])
                i += 1
        
        self.buffer = self.buffer[i:] if i < n else ""
        return "".join(result)

class SimpleGGUFLoader:
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        self.model = None
        self.current_model = None
        self._pending_load_worker = None
        self._cached_total_ram_mb = None
        self._cached_gpu_vram_mb = None
        self._gpu_probe_attempted = False
        self.fast_load_mode = False

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def list_models(self):
        models = []
        if not os.path.exists(self.models_dir):
            return models

        for f in os.listdir(self.models_dir):
            if f.endswith(".gguf") and "mmproj" not in f.lower():
                full = os.path.join(self.models_dir, f)
                try:
                    size = os.path.getsize(full)
                    models.append({
                        "name": f,
                        "path": full,
                        "size_mb": round(size / (1024 * 1024), 2)
                    })
                except:
                    pass

        return sorted(models, key=lambda x: x["name"])

    def _find_mmproj(self, model_name):
        base = os.path.splitext(model_name)[0]
        candidates = [
            f"{base}.mmproj.gguf",
            "mmproj.gguf"
        ]

        for f in os.listdir(self.models_dir):
            if "mmproj" in f.lower() and f.endswith(".gguf"):
                candidates.append(f)

        for c in candidates:
            path = os.path.join(self.models_dir, c)
            if os.path.exists(path):
                return path

        return None

    def _try_load_simple(self, params, timeout_s):
        result = {"model": None, "error": None}
        done = threading.Event()
        started = time.time()
        spinner_proc = None
        spinner_flag = None

        try:
            spinner_flag = os.path.join(
                self.models_dir,
                f".load_spinner_{os.getpid()}_{int(started * 1000)}.flag",
            )
            with open(spinner_flag, "w", encoding="utf-8") as handle:
                handle.write("1")

            spinner_code = (
                "import os,sys,time\n"
                "flag=sys.argv[1]\n"
                "t0=float(sys.argv[2])\n"
                "frames='|/-\\\\'\n"
                "i=0\n"
                "last=0\n"
                "while os.path.exists(flag):\n"
                "    elapsed=max(0.0,time.time()-t0)\n"
                "    line=f'\\r{frames[i%4]} [time: {elapsed:.2f}s]'\n"
                "    sys.stdout.write(line)\n"
                "    sys.stdout.flush()\n"
                "    last=max(last,len(line))\n"
                "    i += 1\n"
                "    time.sleep(0.05)\n"
                "sys.stdout.write('\\r' + (' ' * max(40,last)) + '\\r')\n"
                "sys.stdout.flush()\n"
            )
            spinner_proc = subprocess.Popen(
                [sys.executable, "-u", "-c", spinner_code, spinner_flag, str(started)]
            )
        except Exception:
            spinner_proc = None
            spinner_flag = None

        def worker():
            try:
                llama_cpp = ensure_llama_cpp()
                if llama_cpp is None:
                    result["error"] = "llama-cpp-python not installed"
                    return
                result["model"] = llama_cpp.Llama(**params)
            except Exception as e:
                result["error"] = str(e)
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while not done.wait(0.05):
            if (time.time() - started) >= timeout_s:
                break

        if spinner_flag:
            with contextlib.suppress(Exception):
                os.remove(spinner_flag)
        if spinner_proc is not None:
            with contextlib.suppress(Exception):
                spinner_proc.wait(timeout=0.4)
            if spinner_proc.poll() is None:
                with contextlib.suppress(Exception):
                    spinner_proc.terminate()
                with contextlib.suppress(Exception):
                    spinner_proc.wait(timeout=0.3)

        if not done.is_set():
            return False, f"Timeout > {timeout_s}s"

        if result["error"]:
            return False, result["error"]

        return True, result["model"]

    def _load_model_stable(self, path, try_gpu=True, timeout_s=40, mmproj=None):
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        cpu_threads = max(2, min(8, (os.cpu_count() or 8) // 2))
        raw_ctx = os.environ.get("RUN_AI_N_CTX", "").strip()
        try:
            preferred_ctx = int(raw_ctx) if raw_ctx else 8192
        except Exception:
            preferred_ctx = 8192
        preferred_ctx = max(2048, min(65536, preferred_ctx))
        fast_ctx = preferred_ctx
        cpu_ctx = preferred_ctx

        gpu_params = {
            "model_path": path,
            "n_gpu_layers": -1,
            "n_ctx": fast_ctx,
            "n_batch": 128,
            "n_threads": max(2, (os.cpu_count() or 8) - 1),
            "use_mmap": True,
            "use_mlock": False,
            "verbose": False,
        }

        cpu_params = {
            "model_path": path,
            "n_gpu_layers": 0,
            "n_ctx": cpu_ctx,
            "n_batch": 64,
            "n_threads": cpu_threads,
            "use_mmap": True,
            "use_mlock": False,
            "verbose": False,
        }

        if mmproj:
            gpu_params["mmproj"] = mmproj
            cpu_params["mmproj"] = mmproj

        if try_gpu:
            ok, model = self._try_load_simple(gpu_params, timeout_s)
            if ok:
                return {"ok": True, "model": model}

        ok, model = self._try_load_simple(cpu_params, timeout_s)
        if ok:
            return {"ok": True, "model": model}

        return {"ok": False, "error": model}

    def load(self, model_name, show_try_errors=False):
        llama_cpp = ensure_llama_cpp()
        if llama_cpp is None:
            print("ERROR: llama-cpp-python not installed")
            return False

        if self._pending_load_worker is not None:
            if self._pending_load_worker.is_alive():
                print("ERROR: Previous model load is still running in background.")
                print("   Wait a moment and try again.")
                return False
            self._pending_load_worker = None

        if model_name.isdigit():
            models = self.list_models()
            idx = int(model_name) - 1
            if 0 <= idx < len(models):
                model_name = models[idx]["name"]
            else:
                print("Invalid model number")
                return False

        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            print("Model not found:", model_name)
            return False

        is_vision = any(x in model_name.lower() for x in [
            "llava", "bakllava", "cogvlm", "minicpm-v", "qwen3-vl", "qwen-vl"
        ])

        mmproj = self._find_mmproj(model_name) if is_vision else None
        if is_vision and not mmproj:
            print("Vision model requires mmproj file!")
            return False

        timeout_s = 180
        raw_timeout = os.environ.get("RUN_AI_LOAD_TIMEOUT_S", "").strip()
        if raw_timeout:
            try:
                parsed_timeout = int(raw_timeout)
                if parsed_timeout > 0:
                    timeout_s = max(15, min(1800, parsed_timeout))
            except Exception:
                pass

        try_gpu_env = os.environ.get("RUN_AI_TRY_GPU", "auto").strip().lower()
        if try_gpu_env in {"1", "true", "yes", "on"}:
            try_gpu = True
        elif try_gpu_env in {"0", "false", "no", "off"}:
            try_gpu = False
        else:
            try_gpu = False
            try:
                import torch
                if torch.cuda.is_available():
                    vram_mb = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
                    try_gpu = vram_mb >= max(4096, int(os.path.getsize(model_path) / (1024*1024) * 0.75))
            except Exception:
                try_gpu = False

        load_started = time.time()
        result = self._load_model_stable(
            model_path,
            try_gpu=try_gpu,
            timeout_s=timeout_s,
            mmproj=mmproj
        )

        if result["ok"]:
            self.model = result["model"]
            self.current_model = model_name
            load_elapsed = max(0.0, time.time() - load_started)
            print(f"[time: {load_elapsed:.2f}s]")
            return True

        print("Load failed:", result["error"])
        return False

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                top_p: float = 0.9, stream: bool = False, **kwargs) -> Union[str, None]:
        if not self.model:
            raise ValueError("No model loaded. Use 'load' first.")

        try:
            if stream:
                return self._stream_response(prompt, max_tokens, temperature, top_p, **kwargs)
            response = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error while generating response: {e}")
            raise

    def _stream_response(self, prompt: str, max_tokens: int, temperature: float,
                        top_p: float, **kwargs) -> str:
        try:
            cancel_event = kwargs.pop("cancel_event", None)
            response = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                **kwargs,
            )

            for chunk in response:
                if cancel_event is not None and cancel_event.is_set():
                    break
                token = chunk['choices'][0]['text']
                yield token
        except Exception as e:
            print(f"Error while streaming response: {e}")
            raise

    def _resolve_load_timeout_s(self, file_size_mb: float, is_vision: bool) -> int:
        raw = os.environ.get("RUN_AI_LOAD_TIMEOUT_S", "").strip()
        if raw:
            try:
                parsed = int(raw)
                if parsed > 0:
                    return max(15, min(1800, parsed))
            except Exception:
                pass

        base_s = 45 if is_vision else 30
        per_gb_s = 18 if is_vision else 10
        computed = int(base_s + (max(file_size_mb, 0.0) / 1024.0) * per_gb_s)
        return max(base_s, min(300, computed))

    def _get_total_ram_mb(self) -> int:
        if self._cached_total_ram_mb is not None:
            return self._cached_total_ram_mb
        try:
            import psutil
            self._cached_total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        except:
            self._cached_total_ram_mb = 8192  
        return self._cached_total_ram_mb
    
    def _get_gpu_vram_mb(self) -> int:
        if self._cached_gpu_vram_mb is not None:
            return self._cached_gpu_vram_mb
        if self._gpu_probe_attempted:
            return 0
        self._gpu_probe_attempted = True

        if os.environ.get("RUN_AI_PROBE_GPU", "").strip().lower() not in {"1", "true", "yes"}:
            self._cached_gpu_vram_mb = 0
            return 0

        try:
            import torch
            if torch.cuda.is_available():
                self._cached_gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                return self._cached_gpu_vram_mb
        except:
            pass
        self._cached_gpu_vram_mb = 0
        return 0
    
    def _get_smart_params(self, model_size_mb: float, total_ram_mb: int, gpu_vram_mb: int) -> Optional[Dict]:
        try:
            required_ram = int(model_size_mb * 1.5)

            if gpu_vram_mb > 0 and model_size_mb < gpu_vram_mb * 0.9:
                smart_ctx = min(8192, max(2048, gpu_vram_mb // 4))
                return {
                    "n_gpu_layers": -1,
                    "n_ctx": smart_ctx,
                    "n_batch": 256,
                    "n_threads": 0,
                    "desc": f"Smart: GPU-optimized ({gpu_vram_mb}MB VRAM)",
                }
            elif total_ram_mb > required_ram * 2:
                return {
                    "n_gpu_layers": 0,
                    "n_ctx": 4096,
                    "n_batch": 256,
                    "n_threads": 0,
                    "desc": "Smart: CPU balanced",
                }
            else:
                return {
                    "n_gpu_layers": 0,
                    "n_ctx": 2048,
                    "n_batch": 128,
                    "n_threads": 2,
                    "desc": "Smart: large model, minimal context",
                }
        except Exception as e:
            print(f"Error while selecting parameters: {e}")
            return None

    def unload(self) -> bool:
        if self.model:
            try:
                del self.model
                self.model = None
                self.current_model = None
                import gc
                gc.collect()
                return True
            except Exception as e:
                print(f"Error while unloading model: {e}")
                return False
        return True

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if not self.model:
            return max(1, len(text.split()))
        try:
            payload = text.encode("utf-8", errors="ignore")
            try:
                tokens = self.model.tokenize(payload, add_bos=False)
            except TypeError:
                tokens = self.model.tokenize(payload)
            return max(0, len(tokens))
        except Exception:
            return max(1, len(text.split()))

    def download_model(self, source: str, output_name: Optional[str] = None, overwrite: bool = False) -> Dict[str, Any]:
        import urllib.request
        import urllib.error
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        if source.startswith(('http://', 'https://', 'file://')):
            url = source
            filename = output_name or os.path.basename(urlparse(source).path) or "model.gguf"
        else:
            filename = output_name or source
            if not filename.endswith('.gguf'):
                filename += '.gguf'
            if os.path.exists(source):
                dest = os.path.join(self.models_dir, filename)
                if os.path.exists(dest) and not overwrite:
                    return {
                        "status": "already_exists",
                        "name": filename,
                        "path": dest,
                        "size": os.path.getsize(dest),
                        "sha256": self._sha256_file(dest),
                    }
                shutil.copy2(source, dest)
                return {
                    "status": "downloaded",
                    "name": filename,
                    "path": dest,
                    "size": os.path.getsize(dest),
                    "sha256": self._sha256_file(dest),
                }
            else:
                raise ValueError(f"Source not found: {source}")
        
        destination = os.path.join(self.models_dir, filename)
        
        if os.path.exists(destination) and not overwrite:
            return {
                "status": "already_exists",
                "name": filename,
                "path": destination,
                "size": os.path.getsize(destination),
                "sha256": self._sha256_file(destination),
            }
        
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, destination)
        
        return {
            "status": "downloaded",
            "name": filename,
            "path": destination,
            "size": os.path.getsize(destination),
            "sha256": self._sha256_file(destination),
        }
    
    def _sha256_file(self, filepath: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def list_model_aliases(self):
        return []

class OllamaAPIHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        self._set_headers(200)
        self.wfile.write(b'')
    
    def do_GET(self):
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            if path in ('', '/'):
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "service": "CMD LOCAL AI",
                    "host": f"http://localhost:{HTTP_PORT}",
                    "loaded_model": loader.current_model if loader else None
                }).encode())

            elif path in ('/api/tags', '/api/tags/', '/tags', '/tags/'):
                self._handle_tags()

            elif path.startswith('/api/show') or path.startswith('/show'):
                self._handle_show_model()

            elif path in ('/api/version', '/api/version/', '/version', '/version/'):
                self._handle_version()

            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({
                    "error": f"Endpoint {path} does not exist"
                }).encode())

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "error": f"Internal server error: {str(e)}"
            }).encode())

    def do_POST(self):
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data) if post_data else {}
            except json.JSONDecodeError:
                data = {}

            if path in ('/api/generate', '/generate'):
                self._handle_generate(data)

            elif path in ('/api/chat', '/chat'):
                self._handle_chat(data)

            elif path.startswith('/api/pull') or path.startswith('/pull'):
                self._handle_pull(data)

            elif path.startswith('/api/copy') or path.startswith('/copy'):
                self._handle_copy(data)

            elif path in ('', '/'):
                if isinstance(data, dict) and data.get('messages'):
                    self._handle_chat(data)
                elif isinstance(data, dict) and data.get('prompt'):
                    self._handle_generate(data)
                else:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({
                        "error": "Error: for POST / provide 'prompt' or 'messages'."
                    }).encode())

            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({
                    "error": f"Endpoint {path} does not exist"
                }).encode())

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "error": f"Internal server error: {str(e)}"
            }).encode())

    def _handle_tags(self):
        models = loader.list_models()
        
        response = {
            "models": [
                {
                    "name": m['name'],
                    "modified_at": datetime.now().isoformat() + "Z",
                    "size": int(m['size_mb'] * 1024 * 1024),
                    "digest": hashlib.sha256(m['name'].encode()).hexdigest(),
                    "details": {
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "7B",
                        "quantization_level": "Q4_0"
                    }
                }
                for m in models
            ]
        }
        
        self._set_headers(200)
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def _handle_show_model(self):
        if not loader.current_model:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "No model loaded"
            }).encode())
            return

        model_info = {
            "license": "MIT",
            "modelfile": f"# Modelfile for {loader.current_model}\nFROM {loader.current_model}",
            "parameters": "num_ctx 4096",
            "template": "{{ if .System }}<|system|>\n{{ .System }}<|end|>\n{{ end }}{{ .Prompt }}<|end|>\n<|assistant|>",
            "details": {
                "family": "llama",
                "families": ["llama"],
                "format": "gguf",
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        }

        self._set_headers(200)
        self.wfile.write(json.dumps(model_info, indent=2).encode())

    def _handle_version(self):
        version_info = {
            "version": "0.1.0",
            "compatibility": {
                "ollama": "0.1.0",
                "llama.cpp": "master"
            }
        }
        
        self._set_headers(200)
        self.wfile.write(json.dumps(version_info, indent=2).encode())
    
    def _handle_generate(self, data: Dict):
        if not loader.current_model:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "No model loaded. Use 'load' in terminal."
            }).encode())
            return

        prompt = data.get('prompt', '')
        model = data.get('model', '')
        stream = data.get('stream', False)
        options = data.get('options', {})

        max_tokens = options.get('num_predict', 512)
        temperature = options.get('temperature', 0.7)
        top_p = options.get('top_p', 0.9)

        if model and model != loader.current_model:
            models = loader.list_models()
            found = None
            for m in models:
                if m['name'] == model:
                    found = m
                    break

            if found:
                print(f"[API] Loading model on demand: {model}")
                success = loader.load(found['name'])
                if not success:
                    self._set_headers(500)
                    self.wfile.write(json.dumps({
                        "error": f"Failed to load model: {model}"
                    }).encode())
                    return
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({
                    "error": f"Model '{model}' not found. Available models: {[m['name'] for m in models]}"
                }).encode())
                return

        try:
            if stream:
                self._stream_generate(prompt, max_tokens, temperature, top_p)
            else:
                response = loader.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )

                result = {
                    "model": loader.current_model,
                    "created_at": datetime.now().isoformat() + "Z",
                    "response": response,
                    "done": True,
                    "done_reason": "stop",
                    "context": [],
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": len(prompt.split()),
                    "eval_count": len(response.split()),
                    "eval_duration": 0
                }

                self._set_headers(200)
                self.wfile.write(json.dumps(result, indent=2).encode())

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "error": f"Error while generating response: {str(e)}"
            }).encode())

    def _stream_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        
        try:
            response_generator = loader.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )
            
            full_response = ""
            
            for token in response_generator:
                full_response += token
                
                response_obj = {
                    "model": loader.current_model,
                    "created_at": datetime.now().isoformat() + "Z",
                    "response": token,
                    "done": False
                }
                
                self.wfile.write(f"data: {json.dumps(response_obj)}\n\n".encode())
                self.wfile.flush()
            
            final_response = {
                "model": loader.current_model,
                "created_at": datetime.now().isoformat() + "Z",
                "response": "",
                "done": True,
                "done_reason": "stop",
                "context": [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": len(prompt.split()),
                "eval_count": len(full_response.split()),
                "eval_duration": 0
            }
            
            self.wfile.write(f"data: {json.dumps(final_response)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.wfile.write(f"data: {json.dumps(error_response)}\n\n".encode())

    def _handle_chat(self, data: Dict):
        if not loader.current_model:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "No model loaded. Use 'load' in terminal."
            }).encode())
            return

        try:
            messages = data.get('messages', [])
            stream = data.get('stream', False)
            options = data.get('options', {})

            max_tokens = options.get('num_predict', 512)
            temperature = options.get('temperature', 0.7)
            top_p = options.get('top_p', 0.9)

            prompt = self._format_chat_messages(messages)

            if stream:
                self._stream_chat_response(messages, prompt, max_tokens, temperature, top_p)
            else:
                response = loader.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )

                response_obj = {
                    "model": loader.current_model,
                    "created_at": datetime.now().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": len(prompt.split()),
                    "eval_count": len(response.split()),
                    "eval_duration": 0
                }

                self._set_headers(200)
                self.wfile.write(json.dumps(response_obj, indent=2).encode())

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "error": f"Error while generating response: {str(e)}"
            }).encode())

    def _format_chat_messages(self, messages: List[Dict]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted.append(f"<|system|>\n{content}\n<|end|>")
            elif role == 'user':
                formatted.append(f"<|user|>\n{content}\n<|end|>")
            elif role == 'assistant':
                formatted.append(f"<|assistant|>\n{content}\n<|end|>")
        
        return "\n".join(formatted) + "\n<|assistant|>\n"

    def _stream_chat_response(self, messages: List[Dict], prompt: str, max_tokens: int, 
                            temperature: float, top_p: float):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        
        try:
            if not loader or not loader.model:
                self.wfile.write(b'data: {"error": "No model loaded"}\n\n')
                return
            
            full_response = ""
            for token in loader.generate(prompt, max_tokens, temperature, top_p, stream=True):
                if token:
                    full_response += token
                    response_data = {
                        "model": loader.current_model or "unknown",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "message": {
                            "role": "assistant",
                            "content": full_response
                        },
                        "done": False
                    }
                    self.wfile.write(f'data: {json.dumps(response_data)}\n\n'.encode())
                    self.wfile.flush()
            
            final_data = {
                "model": loader.current_model or "unknown",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": len(full_response.split()) if full_response else 0,
                "eval_duration": 0
            }
            self.wfile.write(f'data: {json.dumps(final_data)}\n\n'.encode())
            self.wfile.flush()
            
        except Exception as e:
            error_data = {"error": f"Streaming error: {str(e)}"}
            self.wfile.write(f'data: {json.dumps(error_data)}\n\n'.encode())
            self.wfile.flush()
        
    def _handle_pull(self, data: Dict):
        source = str(data.get("name") or data.get("model") or data.get("source") or "").strip()
        output_name = data.get("filename") or data.get("output")
        auto_load = bool(data.get("load", False))

        if not source:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "Missing field: provide 'name' (alias/URL)"
            }).encode())
            return

        try:
            result = loader.download_model(
                source=source,
                output_name=output_name,
                overwrite=bool(data.get("overwrite", False)),
            )

            loaded = False
            if auto_load:
                loaded = loader.load(result["name"])

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "success",
                "name": result["name"],
                "path": result["path"],
                "size": result["size"],
                "sha256": result["sha256"],
                "loaded": loaded,
            }).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "error": f"Failed to download model: {e}"
            }).encode())

    def _handle_copy(self, data: Dict):
        self._set_headers(200)
        self.wfile.write(json.dumps({
            "status": "success"
        }).encode())

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def start_http_server(port: int = HTTP_PORT) -> socketserver.TCPServer:
    httpd = None
    selected_port = port
    last_error = None

    for candidate_port in range(port, port + 10):
        try:
            server_address = ('', candidate_port)
            httpd = ReusableTCPServer(server_address, OllamaAPIHandler)
            selected_port = candidate_port
            break
        except OSError as exc:
            last_error = exc
            continue

    if httpd is None:
        raise last_error or OSError("No free port for HTTP server.")

    def server_thread():
        print(f"HTTP server running at http://localhost:{selected_port}")
        print("  - GET  /tags         - list models")
        print("  - POST /generate     - generate text")
        print("  - POST /chat         - chat")
        print("  - POST /pull         - download model (alias/URL)")
        print("  - GET  /version      - API version")
        print("\nPress Ctrl+C to stop the server")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping HTTP server...")
            httpd.shutdown()
            httpd.server_close()

    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    return httpd

def clear_screen():
    stdin_ok = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    stdout_ok = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not (stdin_ok and stdout_ok):
        return
    os.system('cls' if os.name == 'nt' else 'clear')

def _read_terminal_line(prompt: str) -> str:
    stdin_ok = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    stdout_ok = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not (stdin_ok and stdout_ok):
        return input(prompt)

    if os.name != "nt":
        return input(prompt)

    try:
        import msvcrt
    except Exception:
        return input(prompt)

    sys.stdout.write(prompt)
    sys.stdout.flush()
    buffer: List[str] = []

    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(buffer)

        if ch == "\x03":
            raise KeyboardInterrupt

        if ch == "\x1a":
            raise EOFError

        if ch == "\x1b":
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "\x1b"

        if ch in ("\x00", "\xe0"):
            try:
                _ = msvcrt.getwch()
            except Exception:
                pass
            continue

        if ch == "\x08":
            if buffer:
                buffer.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue

        buffer.append(ch)
        sys.stdout.write(ch)
        sys.stdout.flush()

def _consume_escape_keypress() -> bool:
    if os.name != "nt":
        return False
    try:
        import msvcrt
    except Exception:
        return False

    pressed = False
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            try:
                _ = msvcrt.getwch()
            except Exception:
                pass
            continue
        if ch == "\x1b":
            pressed = True
    return pressed

def get_terminal_width():
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except:
        return 80  

def print_welcome():
    clear_screen()
    
    terminal_width = get_terminal_width()
    terminal_width = max(terminal_width, 40)

    ascii_title = r"""
 ██████╗ ███╗   ███╗██████╗  █████╗ ██╗
██╔════╝ ████╗ ████║██╔══██╗██╔══██╗██║
██║      ██╔████╔██║██║  ██║███████║██║
██║      ██║╚██╔╝██║██║  ██║██╔══██║██║
╚██████╗ ██║ ╚═╝ ██║██████╔╝██║  ██║██║
 ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝
"""

    print("\n" + "=" * terminal_width)

    for line in ascii_title.splitlines():
        if line.strip() == "":
            print()
        else:
            padding = (terminal_width - len(line)) // 2
            print(" " * max(padding, 0) + line)

    print("=" * terminal_width)

    server_text = f"   HTTP Server: http://localhost:{HTTP_PORT}"
    print(server_text)

    print("=" * terminal_width)

def _print_help_section(title: str, rows: List[Tuple[str, str]]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print("-" * 60)
    command_width = max(len(cmd) for cmd, _ in rows)
    for command, description in rows:
        print(f"  {command.ljust(command_width)}  {description}")

def show_quick_commands() -> None:
    print("\nQuick commands: models | load | /swap | ai | chat <text> | /pause | help | exit")
    print("Tips: Esc unloads current model (app keeps running).")

def show_help():
    term_width = shutil.get_terminal_size().columns
    
    if term_width < 80:
        term_width = 80
    left_width = int(term_width * 0.45)  
    right_width = int(term_width * 0.45) 
    separator_width = 4  
    
    print("\n" + "=" * term_width)
    print(f"{'HELP':^{term_width}}")
    print("=" * term_width)
    
    models_cmds = [
        ("models", "List local GGUF models"),
        ("load", "Choose model from list"),
        ("load <name>", "Load specific model"),
        ("/swap [name]", "Swap model and keep chat history"),
        ("catalog", "Show downloadable aliases"),
        ("download <url>", "Download model"),
        ("pull <url>", "Alias for download"),
        ("unload", "Unload model from RAM")
    ]
    
    chat_cmds = [
        ("ai", "Start chat mode"),
        ("chat <text>", "Send message"),
        ("/pause", "Pause/resume chat"),
        ("(plain text)", "Send as chat message")
    ]
    
    system_cmds = [
        ("status", "Show app status"),
        ("version", "Show llama-cpp version")
    ]
    if not HAS_AI_ENGINE:
        system_cmds.append(("install", "Install AI engine"))
    else:
        system_cmds.append(("update", "Update llama-cpp"))
    system_cmds.extend([
        ("clear", "Clear terminal"),
        ("help", "Show this help"),
        ("exit", "Exit application")
    ])
    
    server_cmds = [
        (f"HTTP: http://localhost:{HTTP_PORT}", ""),
        ("GET /tags", "List models"),
        ("POST /generate", "Generate text"),
        ("POST /chat", "Chat endpoint"),
        ("POST /pull", "Download model")
    ]
    
    models_centered = f"{'MODELS':^{left_width}}"
    chat_centered = f"{'CHAT':^{right_width}}"
    print(f"\n{models_centered}{' ' * separator_width}{chat_centered}")
    print("-" * term_width)

    max_mc = max(len(models_cmds), len(chat_cmds))
    for i in range(max_mc):
        model_cmd = models_cmds[i] if i < len(models_cmds) else ("", "")
        chat_cmd = chat_cmds[i] if i < len(chat_cmds) else ("", "")

        if model_cmd[0]:
            left = f"{model_cmd[0]:<20} - {model_cmd[1]}"
        else:
            left = ""
        left_formatted = f"{left:<{left_width}}"

        if chat_cmd[0]:
            right = f"{chat_cmd[0]:<20} - {chat_cmd[1]}"
        else:
            right = ""
        right_formatted = f"{right:<{right_width}}"

        print(f"{left_formatted}{' ' * separator_width}{right_formatted}")

    system_centered = f"{'SYSTEM':^{left_width}}"
    server_centered = f"{'SERVER':^{right_width}}"
    print(f"\n{system_centered}{' ' * separator_width}{server_centered}")
    print("-" * term_width)
    
    max_sc = max(len(system_cmds), len(server_cmds))
    for i in range(max_sc):
        system_cmd = system_cmds[i] if i < len(system_cmds) else ("", "")
        server_cmd = server_cmds[i] if i < len(server_cmds) else ("", "")

        if system_cmd[0]:
            left = f"{system_cmd[0]:<20} - {system_cmd[1]}"
        else:
            left = ""
        left_formatted = f"{left:<{left_width}}"

        if server_cmd[0]:
            right = f"{server_cmd[0]:<20} - {server_cmd[1]}"
        else:
            right = ""
        right_formatted = f"{right:<{right_width}}"

        print(f"{left_formatted}{' ' * separator_width}{right_formatted}")
    
    print("=" * term_width)

    if LAST_UPDATE_STATUS is not None:
        print(f"\n   Update status: {LAST_UPDATE_STATUS}")

def show_models_menu():
    if not HAS_AI_ENGINE:
        print("\n" + "=" * 60)
        print("ERROR: AI ENGINE NOT INSTALLED")
        print("=" * 60)
        print("\nRun command: install")
        print("=" * 60)
        return

    models = loader.list_models()

    if not models:
        print("\n" + "=" * 60)
        print("ERROR: NO GGUF MODELS IN FOLDER")
        print("=" * 60)
        print("\n1. Create folder 'models/' if missing")
        print("2. Put .gguf files into ./models/")
        print("3. Restart the program")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("  AVAILABLE GGUF MODELS")
    print("=" * 60)

    for i, model in enumerate(models, 1):
        size_mb = model['size_mb']
        if size_mb > 1024:
            size_str = f"{size_mb/1024:.1f} GB"
        else:
            size_str = f"{size_mb:.0f} MB"
        print(f"\n{i:2d}. {model['name']} ({size_str})")

    print("\n" + "=" * 60)
    print("  Enter model number to load (or 'q' / Esc to cancel):")

def show_download_catalog():
    aliases = loader.list_model_aliases() if loader else []
    print("\n" + "=" * 60)
    print("  DOWNLOADABLE MODEL ALIASES")
    print("=" * 60)
    if not aliases:
        print("  No aliases defined.")
        print("=" * 60)
        return

    for entry in aliases:
        print(f"  {entry['alias']:<28} -> {entry['filename']}")
    print("=" * 60)
    print("Usage: download <alias|url|owner/repo/file.gguf> [file.gguf]")

def show_status():
    print("\n" + "=" * 60)
    print("  SYSTEM STATUS")
    print("=" * 60)

    print("\nSYSTEM:")
    print(f"  OS: {sys.platform}")
    print(f"  Python: {sys.version.split()[0]}")

    print("\nMODEL:")
    if loader and loader.current_model:
        print(f"  LOADED: {loader.current_model}")
        if hasattr(loader, 'model') and loader.model:
            print(f"  Context: {loader.model.n_ctx} tokens")
            print(f"  GPU layers: {loader.model.n_gpu_layers if hasattr(loader.model, 'n_gpu_layers') else 'none'}")
    else:
        print("  No model loaded")

    print("\nSERVER:")
    print(f"  Status: {'RUNNING' if http_server else 'STOPPED'}")
    print(f"  Address: http://localhost:{HTTP_PORT}")
    print("\n" + "=" * 60)

def _reasoning_stop_tokens() -> List[str]:
    return [
        "<think>",
        "</think>",
        "<analysis>",
        "</analysis>",
        "<reasoning>",
        "</reasoning>",
        "```thinking",
        "\nAnalysis:",
        "\nReasoning:",
        "\nThought:",
        "</final>",
        "\nUser question:",
        "\nQuestion:",
    ]

def _build_answer_only_prompt(user_text: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    lines = []
    
    lines.append("<|system|>")
    lines.append("Jesteś pomocnym asystentem AI. Odpowiadaj zwięźle i na temat.")
    lines.append("<|end|>")
    
    if history:
        for msg in history[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"<|user|>")
                lines.append(content)
                lines.append("<|end|>")
            elif role == "assistant":
                lines.append(f"<|assistant|>")
                lines.append(content)
                lines.append("<|end|>")
    
    lines.append("<|user|>")
    lines.append(user_text)
    lines.append("<|end|>")
    lines.append("<|assistant|>")
    
    return "\n".join(lines)

def _run_with_spinner(func, desc: str = "generating", allow_cancel: bool = False, cancel_event: Optional[threading.Event] = None):
    outcome: Dict[str, Any] = {"value": None, "error": None}
    finished = threading.Event()

    def _target():
        try:
            outcome["value"] = func()
        except Exception as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    frames = ["|", "/", "-", "\\"]
    idx = 0
    started = time.time()
    last_len = 0
    cancel_requested = False
    
    while not finished.wait(0.02 if allow_cancel else 0.05):
        if allow_cancel and cancel_event is not None and not cancel_requested and _consume_escape_keypress():
            cancel_event.set()
            cancel_requested = True
            if last_len > 0:
                sys.stdout.write("\r" + " " * last_len + "\r")
                sys.stdout.flush()
                last_len = 0

        if cancel_requested:
            continue

        elapsed = max(0.0, time.time() - started)
        line = f"\r{frames[idx % len(frames)]} [time: {elapsed:.2f}s]"
        sys.stdout.write(line)
        sys.stdout.flush()
        last_len = len(line)
        idx += 1

    sys.stdout.write("\r" + " " * last_len + "\r")
    sys.stdout.flush()

    worker.join(timeout=0.2)
    if outcome["error"] is not None:
        raise outcome["error"]
    return outcome["value"], time.time() - started

def send_terminal_prompt(prompt: str, max_tokens: int = -1, temperature: float = 0.7, top_p: float = 0.9) -> bool:
    global TERMINAL_CHAT_HISTORY
    
    if not loader or not loader.current_model:
        print("ERROR: No model loaded. Use 'load' first.")
        return False

    text = (prompt or "").strip()
    if not text:
        print("ERROR: Empty prompt.")
        return False

    try:
        requested_max_tokens = int(max_tokens) if max_tokens is not None else -1
        if requested_max_tokens == 0:
            requested_max_tokens = -1
        safe_temperature = min(max(float(temperature), 0.0), 0.7)
        safe_top_p = min(max(float(top_p), 0.1), 0.95)

        full_prompt = _build_answer_only_prompt(text, TERMINAL_CHAT_HISTORY)
        
        cancel_event = threading.Event()

        def generate_func():
            chunks: List[str] = []
            token_stream = loader.generate(
                prompt=full_prompt,
                max_tokens=requested_max_tokens,
                temperature=safe_temperature,
                top_p=safe_top_p,
                stream=True,
                stop=_reasoning_stop_tokens(),
                cancel_event=cancel_event,
            )
            for token in token_stream:
                chunks.append(token)
            return "".join(chunks)

        response, elapsed = _run_with_spinner(
            generate_func,
            "generating",
            allow_cancel=True,
            cancel_event=cancel_event,
        )
        total_elapsed = elapsed

        if cancel_event.is_set():
            print("\nINFO: Generation interrupted.")
            return False

        filtered = _extract_visible_answer(response or "")
        
        if not filtered:
            simple_prompt = _build_answer_only_prompt(text, None)
            retry_cancel_event = threading.Event()
            
            def retry_generate():
                chunks: List[str] = []
                token_stream = loader.generate(
                    prompt=simple_prompt,
                    max_tokens=requested_max_tokens,
                    temperature=min(safe_temperature, 0.3),
                    top_p=min(safe_top_p, 0.9),
                    stream=True,
                    stop=_reasoning_stop_tokens(),
                    cancel_event=retry_cancel_event,
                )
                for token in token_stream:
                    chunks.append(token)
                return "".join(chunks)
            
            response, retry_elapsed = _run_with_spinner(
                retry_generate,
                "retrying",
                allow_cancel=True,
                cancel_event=retry_cancel_event,
            )
            total_elapsed += retry_elapsed
            if retry_cancel_event.is_set():
                print("\nINFO: Generation interrupted.")
                return False
            filtered = _extract_visible_answer(response or "")
            
        if not filtered:
            print("\nAI: [brak odpowiedzi]")
            return False

        prompt_tokens = loader.count_tokens(full_prompt)
        output_tokens = loader.count_tokens(filtered)
        total_tokens = int(prompt_tokens) + int(output_tokens)

        print(f"\n[Tokens: {total_tokens} | Time: {total_elapsed:.2f}s]")
        print(f"AI: {filtered}")
        
        TERMINAL_CHAT_HISTORY.append({"role": "user", "content": text})
        TERMINAL_CHAT_HISTORY.append({"role": "assistant", "content": filtered})
        
        return True
        
    except Exception as e:
        print(f"ERROR: Chat generation failed: {e}")
        return False

def _extract_visible_answer(raw_text: str) -> str:
    if not raw_text:
        return ""
    
    text = raw_text
    block_patterns = [
        r"<\s*think\s*>.*?<\s*/\s*think\s*>",
        r"<\s*analysis\s*>.*?<\s*/\s*analysis\s*>",
        r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>",
    ]
    for pattern in block_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    lower_text = text.lower()
    final_marker = "<|channel|>final|>"
    final_pos = lower_text.rfind(final_marker)
    if final_pos != -1:
        text = text[final_pos + len(final_marker):]
        lower_tail = text.lower()
        end_idx = lower_tail.find("<|end|>")
        if end_idx != -1:
            text = text[:end_idx]

    final_match = re.search(
        r"(?is)(?:assistant\s*final|assistantfinal)\s*[:>\-| ]+\s*(.+)$",
        text
    )
    if final_match:
        text = final_match.group(1)

    ai_markers = list(re.finditer(r"(?im)^\s*AI\s*:\s*", text))
    if ai_markers:
        text = text[ai_markers[-1].start():]
        text = re.sub(r"(?im)^\s*AI\s*:\s*", "", text, count=1)
    
    marker_assistant = "<|assistant|>"
    marker_end = "<|end|>"
    marker_user = "<|user|>"
    marker_system = "<|system|>"
    lower_text = text.lower()
    assistant_pos = lower_text.rfind(marker_assistant)
    if assistant_pos != -1:
        text = text[assistant_pos + len(marker_assistant):]
        lower_tail = text.lower()
        cut_positions = []
        for marker in (marker_end, marker_user, marker_system):
            idx = lower_tail.find(marker)
            if idx != -1:
                cut_positions.append(idx)
        if cut_positions:
            text = text[:min(cut_positions)]

    text = re.sub(r"<\s*/\s*assistant\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*assistant\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|assistant[^>]*\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|end\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+?\|>", "", text, flags=re.IGNORECASE)
    return text.strip()

def _is_escape_input(text: str) -> bool:
    value = (text or "").strip().lower()
    return value in {"exit", "quit"} or ("\x1b" in (text or ""))

def _is_chat_pause_toggle(text: str) -> bool:
    value = (text or "").strip().lower()
    return value in {"/pause", ":pause"}

def _is_model_unload_shortcut(text: str) -> bool:
    value = (text or "").strip().lower()
    return value in {"esc", "/esc", ":esc", "q", "/q", ":q"} or ("\x1b" in (text or ""))

def _handle_swap_command(raw_command: str) -> bool:
    global TERMINAL_CHAT_HISTORY
    if not loader:
        print("ERROR: Loader not initialized")
        return False

    parts = (raw_command or "").strip().split(maxsplit=1)
    target_model = parts[1].strip() if len(parts) > 1 else ""

    if target_model:
        preserved_history = list(TERMINAL_CHAT_HISTORY)
        swapped = loader.load(target_model, show_try_errors=True)
        if swapped:
            TERMINAL_CHAT_HISTORY = preserved_history
            print("INFO: Model swapped. Chat history kept.")
        return swapped

    show_models_menu()
    choice = _read_terminal_line("> ").strip()
    if choice.lower() in {"q", "quit", "cancel"}:
        print("INFO: Swap cancelled.")
        return False

    models = loader.list_models()
    selected_name = choice
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            selected_name = models[idx]["name"]
        else:
            print("ERROR: Invalid model number")
            return False

    preserved_history = list(TERMINAL_CHAT_HISTORY)
    swapped = loader.load(selected_name, show_try_errors=True)
    if swapped:
        TERMINAL_CHAT_HISTORY = preserved_history
        print("INFO: Model swapped. Chat history kept.")
    return swapped

def run_terminal_chat_session() -> None:
    global TERMINAL_CHAT_HISTORY
    
    if not loader or not loader.current_model:
        print("ERROR: No model loaded. Use 'load' first.")
        return

    print("\nChat mode started. Press Esc or 'q' to unload model and leave chat.")
    print("Type 'exit' or 'quit' to leave chat without unloading.")
    print("Use '/pause' to pause/resume chat.")
    chat_paused = False
    
    while True:
        try:
            chat_prompt = "> " if chat_paused else "AI> "
            user_text = _read_terminal_line(chat_prompt)
        except KeyboardInterrupt:
            print("\nChat interrupted.")
            break
        except EOFError:
            print("\nChat input stream closed.")
            break

        if _is_chat_pause_toggle(user_text):
            chat_paused = not chat_paused
            if chat_paused:
                print("INFO: Chat paused. Use '/pause' to resume.")
            else:
                print("INFO: Chat resumed.")
            continue

        if (user_text or "").strip().lower().split()[0:1] in [["/swap"], ["swap"]]:
            _handle_swap_command(user_text)
            continue

        if _is_model_unload_shortcut(user_text):
            if loader and loader.current_model:
                if loader.unload():
                    print("SUCCESS: Model unloaded from memory")
                    TERMINAL_CHAT_HISTORY.clear()
                else:
                    print("ERROR: Failed to unload model")
            else:
                print("INFO: No model loaded")
            print("Exited chat mode.")
            break

        if _is_escape_input(user_text):
            print("Exited chat mode.")
            break

        if chat_paused:
            print("INFO: Chat is paused. Use '/pause' to resume.")
            continue

        if not (user_text or "").strip():
            continue
            
        send_terminal_prompt(user_text)

def _should_keep_terminal_open() -> bool:
    value = os.environ.get("RUN_AI_KEEP_OPEN", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}

def _wait_before_terminal_close() -> None:
    if not _should_keep_terminal_open():
        return
    stdin_ok = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    stdout_ok = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not (stdin_ok and stdout_ok):
        return
    try:
        input("\nApp stopped. Press Enter to close terminal...")
    except EOFError:
        pass

def main():
    global http_server, loader, HAS_AI_ENGINE, LAST_UPDATE_STATUS, HTTP_PORT, TERMINAL_CHAT_HISTORY

    loader = SimpleGGUFLoader()

    try:
        import llama_cpp
        HAS_AI_ENGINE = True
    except ImportError:
        HAS_AI_ENGINE = False
        print("Warning: AI engine is not installed. Run 'install' to install it.")

    if HAS_AI_ENGINE and sys.version_info >= (3, 13):
        print("Warning: Python 3.13 + llama-cpp-python can be unstable for some GGUF models.")
        print(r"         Recommended runtime: .\python311\python.exe run.py")

    try:
        http_server = start_http_server(HTTP_PORT)
        if http_server:
            try:
                HTTP_PORT = int(http_server.server_address[1])
            except Exception:
                pass
    except Exception as e:
        print(f"ERROR: Failed to start HTTP server: {e}")
        http_server = None

    print_welcome()
    print("Type 'help' for commands.")

    try:
        main_chat_paused = False
        while True:
            try:
                prompt_label = "AI> " if (loader and loader.current_model and not main_chat_paused) else "> "
                raw_command = _read_terminal_line(f"\n{prompt_label}").strip()
                if not raw_command:
                    continue

                if _is_chat_pause_toggle(raw_command):
                    if loader and loader.current_model:
                        main_chat_paused = not main_chat_paused
                        if main_chat_paused:
                            print("INFO: Chat paused. Use '/pause' to resume.")
                        else:
                            print("INFO: Chat resumed.")
                    else:
                        print("INFO: No model loaded")
                    continue

                if _is_model_unload_shortcut(raw_command):
                    if loader and loader.current_model:
                        if loader.unload():
                            print("SUCCESS: Model unloaded from memory")
                            TERMINAL_CHAT_HISTORY.clear()
                        else:
                            print("ERROR: Failed to unload model")
                    else:
                        print("INFO: No model loaded")
                    main_chat_paused = False
                    continue

                command = raw_command.lower()
                command_name = command.split()[0]

                if command_name in ('exit', 'quit', 'wyjdz'):
                    confirm = input("Are you sure you want to exit? (y/n): ").strip().lower()
                    if confirm == 'y':
                        print("Goodbye!")
                        break

                elif command_name in ('help', 'pomoc', '?'):
                    show_help()

                elif command_name == 'models':
                    show_models_menu()
                    choice = _read_terminal_line("> ").strip()
                    if choice.lower() in {"q", "quit", "cancel"}:
                        print("INFO: Load cancelled.")
                        continue
                    if _is_model_unload_shortcut(choice):
                        if loader and loader.current_model:
                            if loader.unload():
                                print("SUCCESS: Model unloaded from memory")
                                TERMINAL_CHAT_HISTORY.clear()
                            else:
                                print("ERROR: Failed to unload model")
                        else:
                            print("INFO: Load cancelled.")
                        continue
                    if choice.isdigit():
                        models = loader.list_models()
                        idx = int(choice) - 1
                        if 0 <= idx < len(models):
                            loader.load(models[idx]['name'], show_try_errors=True)

                elif command_name == 'load':
                    model_name = raw_command[4:].strip()
                    if model_name:
                        loader.load(model_name, show_try_errors=True)
                    else:
                        show_models_menu()
                        choice = _read_terminal_line("> ").strip()
                        if choice.lower() in {"q", "quit", "cancel"}:
                            print("INFO: Load cancelled.")
                            continue
                        if _is_model_unload_shortcut(choice):
                            if loader and loader.current_model:
                                if loader.unload():
                                    print("SUCCESS: Model unloaded from memory")
                                    TERMINAL_CHAT_HISTORY.clear()
                                else:
                                    print("ERROR: Failed to unload model")
                            else:
                                print("INFO: Load cancelled.")
                            continue
                        if choice.isdigit():
                            models = loader.list_models()
                            idx = int(choice) - 1
                            if 0 <= idx < len(models):
                                loader.load(models[idx]['name'], show_try_errors=True)

                elif command_name in ('/swap', 'swap'):
                    _handle_swap_command(raw_command)

                elif command_name == 'catalog':
                    show_download_catalog()

                elif command_name in ('download', 'pull'):
                    parts = raw_command.split(maxsplit=2)
                    source = parts[1].strip() if len(parts) > 1 else ""
                    output_name = parts[2].strip() if len(parts) > 2 else None
                    
                    if not source:
                        print("Usage: download <alias|url> [file.gguf]")
                        show_download_catalog()
                        continue
                    
                    if not loader:
                        print("ERROR: Loader not initialized")
                        continue

                    try:
                        result = loader.download_model(source, output_name=output_name, overwrite=False)
                        if result['status'] == 'already_exists':
                            print(f"Model already exists: {result['name']}")
                            print(f"   Path: {result['path']}")
                            print(f"   Size: {result['size']} B")
                        else:
                            print(f"Model downloaded: {result['name']}")
                            print(f"   Path: {result['path']}")
                            print(f"   Size: {result['size']} B")
                    except ValueError as e:
                        print(f"Invalid source: {e}")
                        print("Try 'catalog' to see available models")
                    except RuntimeError as e:
                        print(f"Download failed: {e}")
                    except Exception as e:
                        print(f"Error: {e}")

                elif command_name == 'unload':
                    if loader.unload():
                        print("SUCCESS: Model unloaded from memory")
                        TERMINAL_CHAT_HISTORY.clear()
                    else:
                        print("ERROR: Failed to unload model")

                elif command_name == 'chat':
                    message = raw_command[4:].strip()
                    if not message:
                        run_terminal_chat_session()
                    else:
                        if main_chat_paused:
                            print("INFO: Chat is paused. Use '/pause' to resume.")
                        else:
                            send_terminal_prompt(message)

                elif command_name == 'ai':
                    run_terminal_chat_session()

                elif command_name == 'status':
                    show_status()

                elif command_name in ('clear', 'cls', 'wyczysc'):
                    clear_screen()
                    TERMINAL_CHAT_HISTORY.clear()
                    print("INFO: Chat history cleared.")

                elif command_name == 'version':
                    if HAS_AI_ENGINE:
                        import llama_cpp
                        print(f"llama-cpp-python: {llama_cpp.__version__}")
                    else:
                        print("ERROR: AI engine is not installed")

                elif command_name == 'update' and HAS_AI_ENGINE:
                    print(f"Updating runtime packages: {', '.join(RECOMMENDED_RUNTIME_PACKAGES)} ...")
                    try:
                        import subprocess
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "--upgrade", *RECOMMENDED_RUNTIME_PACKAGES]
                        )
                        print("SUCCESS: Updated. Restart the app.")
                        break
                    except Exception as e:
                        print(f"ERROR: Update failed: {e}")

                elif command_name == 'install' and not HAS_AI_ENGINE:
                    print(f"Installing runtime packages: {', '.join(RECOMMENDED_RUNTIME_PACKAGES)} ...")
                    try:
                        import subprocess
                        subprocess.check_call([sys.executable, "-m", "pip", "install", *RECOMMENDED_RUNTIME_PACKAGES])
                        print("SUCCESS: Installed. Restart the app.")
                        break
                    except Exception as e:
                        print(f"ERROR: Install failed: {e}")

                else:
                    if loader and loader.current_model:
                        if main_chat_paused:
                            print("INFO: Chat is paused. Use '/pause' to resume.")
                        else:
                            send_terminal_prompt(raw_command)
                    else:
                        print("ERROR: Unknown command. Type 'help' to show available commands")

            except KeyboardInterrupt:
                print("\nUse 'exit' to close the program")
            except EOFError:
                print("\nInput stream is closed. Keeping terminal open...")
                time.sleep(0.5)
            except Exception as e:
                print(f"ERROR: Exception occurred: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"ERROR: Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if http_server:
            print("\nStopping HTTP server...")
            http_server.shutdown()
            http_server.server_close()
        print("Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _wait_before_terminal_close()
