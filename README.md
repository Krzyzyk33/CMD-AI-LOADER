CMD LOCAL AI

## 🔧 Instructions

EN:
Local AI app (CLI + Ollama-compatible API) for running GGUF models with `llama-cpp-python`.


PL:
Lokalna aplikacja AI (CLI + API kompatybilne z Ollama) do uruchamiania modeli GGUF przez `llama-cpp-python`.

---

===============================English===============================

Quick Start

1. Python 3.11+ required.
2. install gguf model and copy to `models/`
3. Run the application:
`py -3 run.py`
4. Load the model:
`load <filename.gguf>`
5. Talk to the model:
`ai`
or:
`chat <text>`

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

===============================Polish================================

Szybki start

1. Wymagany Python 3.11+.
2.zainstaluj model gguf i skopiuj do `models/`
3. Uruchom aplikacje:
   `py -3 run.py`
4. Zaladuj model:
   `load <nazwa_pliku.gguf>`
5. Rozmawiaj z modelem:
   `ai`
   lub:
   `chat <tekst>`

Komendy CLI

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

API

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

===============================Polish================================

- `models/` i duże lokalne pliki binarne są wykluczane z repozytorium przez `.gitignore`.
- Ten projekt został zaprojektowany do uruchamiania w pełni lokalnych modeli.
- Aplikacja może działąć tylko z plikiem `models/` i z plikiem `run.py`

---

## ⭐ Inspirations

===============================English===============================

This project was inspired by tools like **Ollama** and **LM Studio**, which showed how powerful and accessible local AI models can be.  
I wanted to bring a similar experience directly into the command line — simple, fast, and fully local — without the need for heavy interfaces or external services.  
CMD LOCAL AI is my attempt to create a clean, terminal‑native way to interact with AI models.

===============================Polish================================

Inspiracją dla tego projektu były narzędzia takie jak **Ollama** i **LM Studio**, które pokazały, jak potężne i dostępne mogą być lokalne modele AI.
Chciałem przenieść podobne środowisko bezpośrednio do wiersza poleceń — proste, szybkie i w pełni lokalne — bez potrzeby stosowania rozbudowanych interfejsów ani usług zewnętrznych.
CMD LOCAL AI to moja próba stworzenia przejrzystego, natywnego dla terminala sposobu interakcji z modelami AI.


THANKS FOR READ :)








