CMD LOCAL AI

PL:
Lokalna aplikacja AI (CLI + API kompatybilne z Ollama) do uruchamiania modeli GGUF przez `llama-cpp-python`.

EN:
Local AI app (CLI + Ollama-compatible API) for running GGUF models with `llama-cpp-python`.

---

PL - Szybki start
1. Wymagany Python 3.11+.
2. Uruchom aplikacje:
   `py -3 run.py`
3. Zaladuj model:
   `load <nazwa_pliku.gguf>`
4. Rozmawiaj z modelem:
   `ai`
   lub:
   `chat <tekst>`

PL - Komendy CLI
- `load` - wybor modelu z listy
- `load <name>` - ladowanie modelu po nazwie pliku
- `models` - lista modeli GGUF
- `catalog` - lista aliasow do pobierania
- `download <alias|url> [file.gguf]` - pobieranie modelu
- `pull <alias|url> [file.gguf]` - alias dla `download`
- `ai` - tryb rozmowy (`AI>`)
- `chat <tekst>` - pojedyncza wiadomosc
- `unload` - zwolnienie modelu z RAM
- `status` - status aplikacji
- `version` - wersja `llama-cpp-python`
- `update` - aktualizacja runtime

PL - API
- `GET /tags`
- `POST /generate`
- `POST /chat`
- `POST /pull`
- `GET /version`

---

EN - Quick start
1. Python 3.11+ is required.
2. Run the app:
   `py -3 run.py`
3. Load a model:
   `load <model_file.gguf>`
4. Chat with the model:
   `ai`
   or:
   `chat <text>`

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
