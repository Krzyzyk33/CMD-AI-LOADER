CMD LOCAL AI

PL:
Lokalna aplikacja AI (CLI + API kompatybilne z Ollama) do uruchamiania modeli GGUF przez `llama-cpp-python`.

EN:
Local AI app (CLI + Ollama-compatible API) for running GGUF models with `llama-cpp-python`.

---

===============================English===============================

CLI commands
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

API
- `GET /tags`
- `POST /generate`
- `POST /chat`
- `POST /pull`
- `GET /version`
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

===============================Polish===============================

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

Notes:

===============================English===============================

- `models/` and large local binaries are excluded from the repository by `.gitignore`.
- This project is designed to run fully local models.
- The application can only work with the `models/` file and the `run.py` file

===============================Polish===============================

- `models/` i duże lokalne pliki binarne są wykluczane z repozytorium przez `.gitignore`.
- Ten projekt został zaprojektowany do uruchamiania w pełni lokalnych modeli.
- Aplikacja może działąć tylko z plikiem `models/` i z plikiem `run.py`

THANKS FOR READ :)




