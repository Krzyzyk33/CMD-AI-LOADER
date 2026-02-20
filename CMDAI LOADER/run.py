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
        
            try:
                message = (text or b"").decode("utf-8", errors="ignore").strip()
            except Exception:
                message = ""
            if message and int(level) >= 3:
                sys.stderr.write(message + "\n")

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
        self.fast_load_mode = os.environ.get("RUN_AI_FAST_LOAD", "1").strip().lower() not in {"0", "false", "no"}
        self._pending_load_worker: Optional[threading.Thread] = None
        self._cached_total_ram_mb: Optional[int] = None
        self._cached_gpu_vram_mb: Optional[int] = None
        self._gpu_probe_attempted = False
        self._progress_active = False
        self._progress_pct = 0
        self._progress_last_len = 0
        self._progress_frame = 0
        self._progress_current = 0
        self._progress_total = 1
        self._progress_desc = ""
        self._progress_started_at = 0.0
        self._progress_lock = threading.Lock()
        self._progress_stop_event = None
        self._progress_thread = None
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _print_progress(self, current: int, total: int, desc: str = "") -> None:
        """Display progress bar in console"""
        with self._progress_lock:
            self._progress_current = current
            self._progress_total = total
            self._progress_desc = desc
            self._progress_pct = int((current / max(total, 1)) * 100)
            
            if not self._progress_active:
                self._progress_active = True
                self._progress_started_at = time.time()
                self._progress_stop_event = threading.Event()
                self._progress_thread = threading.Thread(target=self._progress_worker)
                self._progress_thread.daemon = True
                self._progress_thread.start()

    @staticmethod
    def _format_progress_units(value: int) -> str:
        try:
            numeric = float(max(0, int(value)))
        except Exception:
            return str(value)
        if numeric >= 1024 ** 3:
            return f"{numeric / (1024 ** 3):.2f}GB"
        if numeric >= 1024 ** 2:
            return f"{numeric / (1024 ** 2):.1f}MB"
        if numeric >= 1024:
            return f"{numeric / 1024:.1f}KB"
        return str(int(numeric))

    @staticmethod
    def _format_progress_units(value: int) -> str:
        try:
            numeric = float(max(0, int(value)))
        except Exception:
            return str(value)
        if numeric >= 1024 ** 3:
            return f"{numeric / (1024 ** 3):.2f}GB"
        if numeric >= 1024 ** 2:
            return f"{numeric / (1024 ** 2):.1f}MB"
        if numeric >= 1024:
            return f"{numeric / 1024:.1f}KB"
        return str(int(numeric))
    
    def _progress_worker(self) -> None:
        frames = ["|", "/", "-", "\\"]
        stop_event = self._progress_stop_event
        if stop_event is None:
            return

        try:
            while not stop_event.is_set():
                with self._progress_lock:
                    if not self._progress_active:
                        break

                    try:
                        width = shutil.get_terminal_size((80, 20)).columns
                    except Exception:
                        width = 80

                    frame = frames[self._progress_frame % len(frames)]
                    elapsed = int(max(0, time.time() - self._progress_started_at))
                    tail = f"  {self._progress_desc}" if self._progress_desc else ""

                    current = max(0, int(self._progress_current))
                    total = max(1, int(self._progress_total))
                    if total >= 1024 * 1024 or current >= 1024 * 1024:
                        current_txt = self._format_progress_units(current)
                        total_txt = self._format_progress_units(total)
                    else:
                        current_txt = str(current)
                        total_txt = str(total)
                    line = f"   {frame} loading  {elapsed:>4}s  [{current_txt}/{total_txt}]{tail}"

                    if len(line) > width:
                        line = line[:width-1] + "..."

                    print("\r" + " " * self._progress_last_len, end="\r")
                    print(line, end="", flush=True)
                    self._progress_last_len = len(line)

                    self._progress_frame += 1

                time.sleep(0.1)
        except Exception:
         
            pass

        print("\r" + " " * self._progress_last_len, end="\r")

    def _progress_newline(self) -> None:
        thread_to_join = None
        with self._progress_lock:
            if self._progress_stop_event:
                self._progress_stop_event.set()
                if self._progress_thread and self._progress_thread.is_alive():
                    thread_to_join = self._progress_thread

            self._progress_active = False
            self._progress_pct = 0
            self._progress_last_len = 0
            self._progress_frame = 0
            self._progress_current = 0
            self._progress_total = 1
            self._progress_desc = ""
            self._progress_started_at = 0.0

        if thread_to_join is not None:
            thread_to_join.join(timeout=0.2)

    def _load_llama_with_timeout(
        self,
        llama_ctor,
        llama_kwargs: Dict[str, Any],
        timeout_s: int = 45,
        attempt: int = 1,
        total_attempts: int = 1,
        desc: str = "",
    ):
        if self._pending_load_worker is not None and self._pending_load_worker.is_alive():
            raise RuntimeError(
                "Previous model load is still running in background after a timeout. "
                "Wait and try again."
            )

        result: Dict[str, Any] = {"model": None, "error": None}

        def _target():
            try:
                result["model"] = llama_ctor(**llama_kwargs)
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(target=_target, daemon=True)
        self._pending_load_worker = worker
        worker.start()

        total_units = max(1, total_attempts * 100)
        started = time.time()
        while worker.is_alive():
            elapsed = time.time() - started
            if elapsed >= max(timeout_s, 1):
                break
           # self._print_progress(
            #     attempt,
            #     max(1, total_attempts),
            #     f"{desc} ({int(elapsed)}s/{int(max(timeout_s, 1))}s)"
            # )
            worker.join(timeout=0.08)

        if worker.is_alive():
            raise TimeoutError(
                f"Model loading timeout exceeded ({timeout_s}s). "
                "Aborting retries to avoid overlapping background loads."
            )
        self._pending_load_worker = None
        if result["error"] is not None:
            raise result["error"]
        return result["model"], time.time() - started

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        if not os.path.exists(self.models_dir):
            return models
            
        for f in os.listdir(self.models_dir):
            if f.endswith('.gguf') and 'mmproj' not in f.lower():
                try:
                    full_path = os.path.join(self.models_dir, f)
                    size = os.path.getsize(full_path)
                    size_mb = size / (1024 * 1024)
                    models.append({
                        'name': f,
                        'size': size,
                        'size_mb': round(size_mb, 2),
                        'path': full_path
                    })
                except:
                    continue
        
        return sorted(models, key=lambda x: x['name'])

    def list_model_aliases(self) -> List[Dict[str, str]]:
       
        aliases: List[Dict[str, str]] = []
        model_aliases = load_model_aliases()
        for alias_name, alias_data in sorted(model_aliases.items()):
            aliases.append({
                "alias": alias_name,
                "filename": alias_data.get("filename", ""),
                "url": alias_data.get("url", ""),
            })
        return aliases

    def _sanitize_filename(self, name: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        cleaned = "".join(ch if ch in allowed else "_" for ch in name).strip("._")
        if not cleaned:
            cleaned = f"model_{int(time.time())}.gguf"
        if not cleaned.lower().endswith(".gguf"):
            cleaned = f"{cleaned}.gguf"
        return cleaned

    def _resolve_download_target(self, source: str, output_name: Optional[str] = None) -> Tuple[str, str]:
        normalized = (source or "").strip()
        normalized_lc = normalized.lower()
        if not normalized:
            raise ValueError("Nie podano zrodla modelu.")

        model_aliases = load_model_aliases()
        
        if normalized in model_aliases:
            url = model_aliases[normalized]["url"]
            default_name = model_aliases[normalized]["filename"]
        elif normalized_lc in model_aliases:
            url = model_aliases[normalized_lc]["url"]
            default_name = model_aliases[normalized_lc]["filename"]
        elif len(normalized) >= 3:
            matches = [k for k in model_aliases.keys() if k.startswith(normalized_lc)]
            if len(matches) == 1:
                chosen = matches[0]
                url = model_aliases[chosen]["url"]
                default_name = model_aliases[chosen]["filename"]
            elif len(matches) > 1:
                raise ValueError(f"Alias niejednoznaczny '{normalized}'. Mozliwe: {', '.join(sorted(matches))}")
            else:
                url = ""
                default_name = ""
        else:
            url = ""
            default_name = ""

        if not url and normalized_lc.endswith(".gguf"):
            model_aliases = load_model_aliases()
            file_matches = [
                k for k, v in model_aliases.items()
                if v.get("filename", "").lower() == normalized_lc
            ]
            if len(file_matches) == 1:
                chosen = file_matches[0]
                url = model_aliases[chosen]["url"]
                default_name = model_aliases[chosen]["filename"]

        if url:
            resolved_name = self._sanitize_filename(output_name or default_name)
            return url, resolved_name

        if normalized.startswith("http://") or normalized.startswith("https://"):
            parsed = urlparse(normalized)
            basename = os.path.basename(parsed.path)
            default_name = unquote(basename) if basename else f"model_{int(time.time())}.gguf"
            url = normalized
            resolved_name = self._sanitize_filename(output_name or default_name)
            return url, resolved_name

        if normalized.count("/") >= 2 and normalized.lower().endswith(".gguf"):
            parts = normalized.split("/")
            owner = parts[0]
            repo = parts[1]
            file_in_repo = "/".join(parts[2:])
            url = f"https://huggingface.co/{owner}/{repo}/resolve/main/{file_in_repo}?download=true"
            resolved_name = self._sanitize_filename(output_name or os.path.basename(file_in_repo))
            return url, resolved_name

        model_aliases = load_model_aliases()
        if model_aliases:
            available = ", ".join(sorted(model_aliases.keys()))
            raise ValueError(
                f"Nieznane zrodlo '{normalized}'. Uzyj aliasu, URL lub owner/repo/file.gguf. Dostepne aliasy: {available}"
            )
        raise ValueError(
            f"Nieznane zrodlo '{normalized}'. Uzyj URL lub owner/repo/file.gguf."
        )

    def _sha256_file(self, file_path: str) -> str:
        """Wylicza SHA-256 dla pobranego pliku."""
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def download_model(
        self,
        source: str,
        output_name: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        url, filename = self._resolve_download_target(source, output_name)
        destination = os.path.join(self.models_dir, filename)
        partial = f"{destination}.part"

        if os.path.exists(destination) and not overwrite:
            return {
                "status": "already_exists",
                "name": filename,
                "path": destination,
                "size": os.path.getsize(destination),
                "sha256": self._sha256_file(destination),
            }

        if os.path.exists(partial):
            try:
                os.remove(partial)
            except Exception:
                pass

        bytes_read = 0
        total_bytes = 0
        print(f"Loading model: {filename}")
        print(f"Source: {source}")

        try:
            request = Request(
                url,
                headers={
                    "User-Agent": f"run.py/{VERSION}",
                    "Accept": "*/*",
                },
            )
            with urlopen(request, timeout=60) as response:
                content_len = response.headers.get("Content-Length", "0")
                try:
                    total_bytes = int(content_len)
                except Exception:
                    total_bytes = 0

                with open(partial, "wb") as output:
                    while True:
                        block = response.read(1024 * 1024 * 2)
                        if not block:
                            break
                        output.write(block)
                        bytes_read += len(block)
                        # self._print_progress(
                        #     bytes_read,
                        #     total_bytes if total_bytes > 0 else max(bytes_read, 1),
                        #     f"download {filename}"
                        # )
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Connection error: {exc}") from exc
        finally:
            self._progress_newline()

        if bytes_read == 0:
            if os.path.exists(partial):
                os.remove(partial)
            raise RuntimeError("Downloaded 0 bytes.")

        os.replace(partial, destination)

        with open(destination, "rb") as model_file:
            header = model_file.read(4)
        if header != b"GGUF":
            os.remove(destination)
            raise RuntimeError("Downloaded file is not a valid GGUF model.")

        return {
            "status": "downloaded",
            "name": filename,
            "path": destination,
            "size": os.path.getsize(destination),
            "sha256": self._sha256_file(destination),
            "url": url,
        }

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
    
    def load(self, model_name: str, show_try_errors: bool = False) -> bool:
        import os
        llama_cpp = ensure_llama_cpp()
        if llama_cpp is None:
            print("ERROR: llama-cpp-python is not installed. Run: install")
            return False

        if self._pending_load_worker is not None:
            if self._pending_load_worker.is_alive():
                print("ERROR: Previous model load is still finishing in background.")
                print("   Wait 10-20 seconds and try again.")
                return False
            self._pending_load_worker = None

        selector = (model_name or "").strip()
        if selector.isdigit():
            models = self.list_models()
            idx = int(selector) - 1
            if 0 <= idx < len(models):
                model_name = models[idx]["name"]
            else:
                print(f"ERROR: Invalid model number: {selector}")
                return False

        if "mmproj" in model_name.lower():
            print("ERROR: mmproj is an add-on for vision models and cannot be loaded alone.")
            return False

        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"ERROR: File {model_name} does not exist in {self.models_dir}/")
            return False

        try:
            file_size = os.path.getsize(model_path)
            file_size_mb = file_size / (1024 * 1024)
            model_name_lc = model_name.lower()

            with open(model_path, 'rb') as f:
                header = f.read(4)
                if header != b'GGUF':
                    print(f"ERROR: {model_name} is not a valid GGUF file")
                    return False

            vision_markers = ('llava', 'bakllava', 'cogvlm', 'minicpm-v', 'qwen3-vl', 'qwen-vl')
            is_vision = any(marker in model_name_lc for marker in vision_markers)
            mmproj_path = self._find_mmproj(model_name) if is_vision else None
            if is_vision and not mmproj_path:
                print("ERROR: Vision model detected but mmproj file was not found.")
                print("   Download mmproj and place it in ./models/")
                print("   Hint: use alias 'qwen3-vl-4b-mmproj'")
                return False

            param_combinations = []

            if file_size > 10 * 1024 * 1024 * 1024:
                param_combinations = [
                    {"n_gpu_layers": -1, "n_ctx": 1024, "n_batch": 128, "n_threads": 0, "desc": "GPU + CPU, fastest init"},
                    {"n_gpu_layers": 0, "n_ctx": 1024, "n_batch": 128, "n_threads": 0, "desc": "CPU, safe load"},
                    {"n_gpu_layers": 0, "n_ctx": 1024, "n_batch": 128, "n_threads": 2, "desc": "CPU, minimal context"},
                ]
            elif file_size > 4 * 1024 * 1024 * 1024:
                param_combinations = [
                    {"n_gpu_layers": -1, "n_ctx": 2048, "n_batch": 128, "n_threads": 0, "desc": "GPU, fastest init"},
                    {"n_gpu_layers": -1, "n_ctx": 1024, "n_batch": 96, "n_threads": 0, "desc": "GPU, reduced context"},
                    {"n_gpu_layers": 0, "n_ctx": 2048, "n_batch": 128, "n_threads": 0, "desc": "CPU fallback"},
                ]
            elif file_size > 1 * 1024 * 1024 * 1024:
                param_combinations = [
                    {"n_gpu_layers": -1, "n_ctx": 2048, "n_batch": 128, "n_threads": 0, "desc": "GPU, fastest init"},
                    {"n_gpu_layers": -1, "n_ctx": 1024, "n_batch": 96, "n_threads": 0, "desc": "GPU, reduced context"},
                    {"n_gpu_layers": 20, "n_ctx": 1024, "n_batch": 96, "n_threads": 0, "desc": "Memory saving"},
                ]
            else:
                param_combinations = [
                    {"n_gpu_layers": -1, "n_ctx": 2048, "n_batch": 128, "n_threads": 0, "desc": "Fast generation"},
                    {"n_gpu_layers": -1, "n_ctx": 1024, "n_batch": 96, "n_threads": 0, "desc": "Very fast generation"},
                    {"n_gpu_layers": 0, "n_ctx": 2048, "n_batch": 128, "n_threads": 0, "desc": "CPU fallback"},
                ]

            if is_vision:
                param_combinations.insert(0, {
                    "n_gpu_layers": -1,
                    "n_ctx": 2048,
                    "n_batch": 128,
                    "n_threads": 0,
                    "desc": "Vision fast init",
                    "mmproj": mmproj_path,
                })
                param_combinations.append({
                    "n_gpu_layers": 0,
                    "n_ctx": 1024,
                    "n_batch": 96,
                    "n_threads": 0,
                    "desc": "Vision CPU fallback",
                    "mmproj": mmproj_path,
                })

            while len(param_combinations) < 3:
                last = param_combinations[-1].copy()
                last["n_ctx"] = max(512, last.get("n_ctx", 2048) // 2)
                last["n_batch"] = max(64, last.get("n_batch", 256) // 2)
                last["n_threads"] = min(8, (last.get("n_threads", 0) or 1) * 2)
                last["desc"] = f"Fallback {len(param_combinations) - 2}"
                param_combinations.append(last)

            if not self.fast_load_mode:
                total_ram = self._get_total_ram_mb()
                gpu_vram = self._get_gpu_vram_mb()
                smart_params = self._get_smart_params(file_size_mb, total_ram, gpu_vram)
                if smart_params:
                    param_combinations.insert(0, smart_params)

            deduped: List[Dict[str, Any]] = []
            seen = set()
            for params in param_combinations:
                key = (
                    params.get("n_gpu_layers"),
                    params.get("n_ctx"),
                    params.get("n_batch"),
                    params.get("n_threads"),
                    params.get("mmproj"),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(params)

            max_attempts = 2 if is_vision else 3
            param_combinations = deduped[:max_attempts]

            if self.model and self.current_model == model_name:
                print(f"Model {model_name} is already loaded.")
                return True

            if self.model:
                self.unload()

            total_attempts = len(param_combinations)
            timeout_s = self._resolve_load_timeout_s(file_size_mb, is_vision)
            last_error = "unknown error"
            timed_out = False
            for attempt, params in enumerate(param_combinations, 1):
                desc = params.get('desc', f"Attempt {attempt}/{total_attempts}")
                # self._print_progress(attempt, total_attempts, desc)

                try:
                    llama_kwargs = {
                        'model_path': model_path,
                        'n_ctx': params.get('n_ctx', 2048),
                        'n_batch': params.get('n_batch', 128),
                        'n_threads': max(1, int(params.get('n_threads') or max(1, (os.cpu_count() or 4) - 1))),
                        'n_gpu_layers': params.get('n_gpu_layers', 0),
                        'use_mmap': True,
                        'use_mlock': False,
                        'verbose': False,
                    }

                    if 'mmproj' in params and params['mmproj']:
                        llama_kwargs['mmproj'] = params['mmproj']
                        llama_kwargs['n_ctx'] = min(llama_kwargs['n_ctx'], 4096)

                    import time
                    spinner_chars = ['|', '/', '-', '\\']
                    start_time = time.time()
                    
                    # Start spinner in background
                    import threading
                    import sys
                    import os
                    spinner_active = True
                    
                    def spinner_thread():
                        while spinner_active:
                            current_time = time.time() - start_time
                            spinner = spinner_chars[int(current_time * 10) % 4]
                            print(f"\r{spinner} [time: {current_time:.1f}s]", end="", flush=True)
                            time.sleep(0.1)
                    
                    spinner_t = threading.Thread(target=spinner_thread, daemon=True)
                    spinner_t.start()
                    
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    
                    class DevNull:
                        def write(self, text):
                            pass
                        def flush(self):
                            pass
                    
                    sys.stdout = DevNull()
                    sys.stderr = DevNull()
                    
                    try:
                    
                        self.model, elapsed = self._load_llama_with_timeout(
                            llama_cpp.Llama,
                            llama_kwargs,
                            timeout_s=timeout_s,
                            attempt=attempt,
                            total_attempts=total_attempts,
                            desc=desc,
                        )
                        self.current_model = model_name
                    finally:
                        
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                    
                    spinner_active = False
                    time.sleep(0.2)  
                    
                    print("\r" + " " * 50 + "\r", end="")  
                    self._progress_newline()
                    print(f"  [time: {elapsed:.2f} sek]")
                    
                    return True

                except TimeoutError as e:
                    last_error = str(e)
                    timed_out = True
                    if show_try_errors:
                        self._progress_newline()
                        print(f"     ERROR: Attempt timed out: {e}")
                    break
                except Exception as e:
                    last_error = str(e)
                    normalized_error = last_error.lower()
                    is_deterministic_load_error = (
                        "failed to load model from file" in normalized_error
                        or "unknown model architecture" in normalized_error
                        or "unsupported model architecture" in normalized_error
                    )
                    if is_deterministic_load_error:
                        if show_try_errors:
                            self._progress_newline()
                            print(f"     ERROR: Attempt failed: {e}")
                        break
                    if show_try_errors:
                        self._progress_newline()
                        print(f"     ERROR: Attempt failed: {e}")
                        if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                            import gc
                            gc.collect()
                            print("     INFO: Garbage collection executed...")

            self._progress_newline()
            print(f"ERROR: Failed to load model {model_name} after {total_attempts} attempts.")
            print(f"   Last error: {last_error}")
            if timed_out:
                print(f"   Hint: loading timeout was {timeout_s}s.")
                print("   Hint: set RUN_AI_LOAD_TIMEOUT_S=120 (or higher for very large models).")
            if "failed to load model from file" in last_error.lower():
                print("   Hint: run 'update' to refresh llama-cpp-python.")
                if is_vision:
                    print("   Hint: vision model support depends on llama-cpp-python build/version.")
            return False

        except Exception as e:
            self._progress_newline()
            print(f"ERROR: Critical error while loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_mmproj(self, model_name: str) -> Optional[str]:
        """Znajdz plik mmproj dla modelu wizyjnego"""
        model_full_path = os.path.join(self.models_dir, model_name)
        model_dir = os.path.dirname(model_full_path) or self.models_dir
        model_stem = os.path.splitext(os.path.basename(model_name))[0]
        mmproj_candidates = [
            os.path.join(model_dir, f"{model_stem}.mmproj.gguf"),
            os.path.join(model_dir, "mmproj.gguf"),
        ]

        try:
            for entry in os.listdir(model_dir):
                entry_lc = entry.lower()
                if entry_lc.endswith(".gguf") and "mmproj" in entry_lc:
                    mmproj_candidates.append(os.path.join(model_dir, entry))
        except Exception:
            pass

        for candidate in mmproj_candidates:
            if os.path.exists(candidate):
                return candidate
        return None

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
            response = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                **kwargs,
            )

            full_response = ""
            for chunk in response:
                token = chunk['choices'][0]['text']
                full_response += token
                yield token
        except Exception as e:
            print(f"Error while streaming response: {e}")
            raise

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
        """Handle POST requests."""
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
                    "size": m['size'],
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
        """Returns model details (Ollama /api/show compatible)."""
        if not loader.current_model:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                "error": "No model loaded"
            }).encode())
            return

        model_info = {
            "license": "Own",
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
        """Handle chat (Ollama /api/chat compatible)."""
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
        """Kopiowanie modelu (kompatybilne z Ollama /api/copy)"""
        self._set_headers(200)
        self.wfile.write(json.dumps({
            "status": "success"
        }).encode())

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_http_server(port: int = HTTP_PORT) -> socketserver.TCPServer:
    """Start HTTP server in a background thread."""
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


def print_welcome():
    clear_screen()
    print("\n" + "=" * 60)
    print("                 CMD LOCAL AI")
    print("=" * 60)
    print(f"   HTTP Server: http://localhost:{HTTP_PORT}")
    print("=" * 60)


def _print_help_section(title: str, rows: List[Tuple[str, str]]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print("-" * 60)
    command_width = max(len(cmd) for cmd, _ in rows)
    for command, description in rows:
        print(f"  {command.ljust(command_width)}  {description}")


def show_quick_commands() -> None:
    print("\nQuick commands: models | load | ai | chat <text> | /pause | help | exit")
    print("Tips: Esc unloads current model (app keeps running).")


def show_help():
    """Show available commands."""
    
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
    """Show model selection menu."""
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
    print("  Enter model number to load (or 'q' to cancel):")


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
        if hasattr(loader, 'model'):
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


def _build_answer_only_prompt(user_text: str) -> str:
    return (
        "You are a helpful assistant.\n"
        "Return only the final answer for the user.\n"
        "Do not output analysis, reasoning, thought process, steps, or hidden thinking.\n"
        "Output format must be exactly: <final>YOUR ANSWER</final>\n\n"
        f"User question:\n{user_text}\n\n"
        "<final>"
    )


def _write_chat_debug(stage: str, user_text: str, raw_response: str, filtered: str) -> None:
    try:
        stamp = datetime.now().isoformat()
        entry = (
            f"[{stamp}] stage={stage}\n"
            f"USER: {user_text}\n"
            f"RAW: {(raw_response or '').replace(chr(13), ' ').replace(chr(10), '\\n')[:3000]}\n"
            f"FILTERED: {(filtered or '').replace(chr(13), ' ').replace(chr(10), '\\n')[:1000]}\n"
            f"{'-' * 80}\n"
        )
        with open("chat_debug.log", "a", encoding="utf-8") as handle:
            handle.write(entry)
    except Exception:
        pass


def _run_with_spinner(func, desc: str = "generating"):
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
    while not finished.wait(0.1):
        elapsed = int(max(0, time.time() - started))
        line = f"{frames[idx % len(frames)]} {desc} {elapsed}s"
        print("\r" + " " * last_len, end="\r")
        print(line, end="", flush=True)
        last_len = len(line)
        idx += 1

    if last_len:
        print("\r" + " " * last_len, end="\r", flush=True)

    worker.join(timeout=0.2)
    if outcome["error"] is not None:
        raise outcome["error"]
    return outcome["value"], time.time() - started


def send_terminal_prompt(prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> bool:
    if not loader or not loader.current_model:
        print("ERROR: No model loaded. Use 'load' first.")
        return False

    text = (prompt or "").strip()
    if not text:
        print("ERROR: Empty prompt.")
        return False

    try:
        env_chat_limit = os.environ.get("RUN_AI_CHAT_MAX_TOKENS", "").strip()
        try:
            default_limit = max(64, min(4096, int(env_chat_limit))) if env_chat_limit else 768
        except Exception:
            default_limit = 768
        capped_max_tokens = max(64, min(int(max_tokens or default_limit), default_limit))
        safe_temperature = min(max(float(temperature), 0.0), 0.7)
        safe_top_p = min(max(float(top_p), 0.1), 0.95)
        total_elapsed = 0.0

        primary_prompt = _build_answer_only_prompt(text)
        response, elapsed = _run_with_spinner(
            lambda: loader.generate(
                prompt=primary_prompt,
                max_tokens=capped_max_tokens,
                temperature=safe_temperature,
                top_p=safe_top_p,
                stream=False,
                stop=_reasoning_stop_tokens(),
            ),
            desc="generating"
        )
        total_elapsed += elapsed

        used_prompt = primary_prompt
        used_raw_response = response or ""
        filtered = _extract_visible_answer(used_raw_response)
        if not filtered:
            _write_chat_debug("primary-empty", text, used_raw_response, filtered)
            retry_prompt = (
                "Answer the question directly.\n"
                "Output only final user-facing answer.\n"
                "Do not include analysis or reasoning.\n"
                "Format: <final>YOUR ANSWER</final>\n"
                f"Question: {text}\n"
                "<final>"
            )
            retry_response, retry_elapsed = _run_with_spinner(
                lambda: loader.generate(
                    prompt=retry_prompt,
                    max_tokens=max(96, min(512, capped_max_tokens)),
                    temperature=min(safe_temperature, 0.3),
                    top_p=min(safe_top_p, 0.9),
                    stream=False,
                    stop=_reasoning_stop_tokens(),
                ),
                desc="retry"
            )
            total_elapsed += retry_elapsed
            used_prompt = retry_prompt
            used_raw_response = retry_response or ""
            filtered = _extract_visible_answer(used_raw_response)
        if not filtered:
            _write_chat_debug("retry-empty", text, used_raw_response, filtered)
            print("\nAI: [brak finalnej odpowiedzi]")
            return False

        prompt_tokens = loader.count_tokens(used_prompt) if hasattr(loader, "count_tokens") else max(1, len(used_prompt.split()))
        output_tokens = loader.count_tokens(filtered) if hasattr(loader, "count_tokens") else max(1, len(filtered.split()))
        total_tokens = int(prompt_tokens) + int(output_tokens)

        print(f"\n[tokens: {total_tokens} | time: {total_elapsed:.2f} sek]")
        print(f"AI: {filtered}")
        return True
    except Exception as e:
        _write_chat_debug("exception", text, "", str(e))
        print(f"ERROR: Chat generation failed: {e}")
        return False


def _extract_visible_answer(raw_text: str) -> str:
    text = OutputFilter().feed(raw_text or "")
    text = text.strip()
    if not text:
        return ""

    final_block = re.search(r"(?is)<\s*final\s*>(.*?)<\s*/\s*final\s*>", text)
    if final_block:
        candidate = final_block.group(1).strip(" \t\"'`.")
        if candidate:
            return candidate

    final_open_only = re.search(r"(?is)<\s*final\s*>(.*)$", text)
    if final_open_only:
        candidate = final_open_only.group(1).strip(" \t\"'`.")
        if candidate:
            return candidate

    assistant_final = re.search(r"(?is)assistant\s*final\s*[:\-]?\s*(.+)$", text)
    if assistant_final:
        candidate = assistant_final.group(1).strip(" \t\"'`.")
        if candidate:
            return candidate

    block_patterns = [
        r"<\s*think\s*>.*?<\s*/\s*think\s*>",
        r"<\s*analysis\s*>.*?<\s*/\s*analysis\s*>",
        r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>",
    ]
    for pattern in block_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.strip()

    marker_regex = re.compile(
        r"(?im)\b(?:final(?:\s+answer)?|answer|odpowiedz)\s*:\s*([^\n\r]+)"
    )
    marker_matches = list(marker_regex.finditer(text))
    if marker_matches:
        candidate = marker_matches[-1].group(1).strip(" \t\"'`.")
        if candidate:
            return candidate

    cleaned_lines: List[str] = []
    reasoning_prefixes = (
        "analysis:",
        "reasoning:",
        "thinking:",
        "thought:",
        "let's think",
        "i think",
        "i should",
        "i will",
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(reasoning_prefixes):
            continue
        if lower.startswith(("so final:", "so answer:", "final:", "answer:")):
            parts = stripped.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                cleaned_lines.append(parts[1].strip())
            continue
        cleaned_lines.append(stripped)

    if not cleaned_lines:
        return text

    deduped: List[str] = []
    for line in cleaned_lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return "\n".join(deduped).strip()


def _is_escape_input(text: str) -> bool:
    value = (text or "").strip().lower()
    return value in {"exit", "quit"} or ("\x1b" in (text or ""))


def _is_chat_pause_toggle(text: str) -> bool:
    value = (text or "").strip().lower()
    return value in {"/pause", ":pause"}


def _is_model_unload_shortcut(text: str) -> bool:
    value = (text or "").strip().lower()
    return value in {"/esc", ":esc"} or ("\x1b" in (text or ""))


def run_terminal_chat_session() -> None:
    """Interactive chat loop with a loaded model."""
    if not loader or not loader.current_model:
        print("ERROR: No model loaded. Use 'load' first.")
        return

    print("\nChat mode started. Press Esc to unload model and leave chat.")
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

        if _is_model_unload_shortcut(user_text):
            if loader and loader.current_model:
                if loader.unload():
                    print("SUCCESS: Model unloaded from memory")
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
    """Main program entrypoint."""
    global http_server, loader, HAS_AI_ENGINE, LAST_UPDATE_STATUS, HTTP_PORT

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
                    choice = input("> ").strip()
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
                        choice = input("> ").strip()
                        if choice.isdigit():
                            models = loader.list_models()
                            idx = int(choice) - 1
                            if 0 <= idx < len(models):
                                loader.load(models[idx]['name'], show_try_errors=True)

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


