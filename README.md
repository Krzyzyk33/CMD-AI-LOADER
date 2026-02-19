CMD LOCAL AI

PL:
Lokalna aplikacja AI (CLI + API kompatybilne z Ollama) do uruchamiania modeli GGUF przez `llama-cpp-python`.

EN:
Local AI app (CLI + Ollama-compatible API) for running GGUF models with `llama-cpp-python`.


EN - CLI commands
- `load` - choose model from list
- `load <name>` - load model by filename
- `models` - list local GGUF models
- `catalog` - list downloadable aliases
- `download <alias|url> [file.gguf]` - download model
- `pull <alias|url> [file.gguf]` - alias for `download`
- `ai` - chat mode (`AI>`)
- `chat <text>` - single message
- `unload` - unload model from RAM
- `status` - app status
- `version` - `llama-cpp-python` version
- `update` - runtime update

EN - API
- `GET /tags`
- `POST /generate`
- `POST /chat`
- `POST /pull`
- `GET /version`

---

Notes:
- `models/` and large local binaries are excluded from the repository by `.gitignore`.
- This project is designed to run fully local models.

