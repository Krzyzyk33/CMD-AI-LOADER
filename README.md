CMD LOCAL AI

## 📝 Short Description

### English:
Local AI app (CLI + Ollama-compatible API) for running GGUF models with `llama-cpp-python`.


### Polish:
Lokalna aplikacja AI (CLI + API kompatybilne z Ollama) do uruchamiania modeli GGUF przez `llama-cpp-python`.

## 🔧 Instructions

## English

### Quick Start

1. Python 3.11+ required.
2. Install a GGUF model and copy it to models/.
3. Run the application:
`py -3 run.py`
4. Load the model:
`load <filename.gguf>`
5. Talk to the model:
`ai`
or:
`chat <text>`

### CLI commands

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

### API

- `GET /tags`
- `POST /generate`
- `POST /chat`
- `POST /pull`
- `GET /version`

## Polish

### Szybki start

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

### Komendy CLI

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

### API

- `GET /tags`
- `POST /generate`
- `POST /chat`
- `POST /pull`
- `GET /version`

---

## Notes:

### English

- `models/` and large local binaries are excluded from the repository by `.gitignore`.
- This project is designed to run fully local models.
- The application can only work with the `models/` file and the `run.py` file

### Polish

- `models/` i duże lokalne pliki binarne są wykluczane z repozytorium przez `.gitignore`.
- Ten projekt został zaprojektowany do uruchamiania w pełni lokalnych modeli.
- Aplikacja działa z katalogiem models/ i plikiem run.py.

---

## ⭐ Inspirations

### English

This project was inspired by tools like **Ollama** and **LM Studio**, which showed how powerful and accessible local AI models can be.  
I wanted to bring a similar experience directly into the command line — simple, fast, and fully local — without the need for heavy interfaces or external services.  
CMD LOCAL AI is my attempt to create a clean, terminal‑native way to interact with AI models.

### Polish

Inspiracją dla tego projektu były narzędzia takie jak **Ollama** i **LM Studio**, które pokazały, jak potężne i dostępne mogą być lokalne modele AI.
Chciałem przenieść podobne środowisko bezpośrednio do wiersza poleceń — proste, szybkie i w pełni lokalne — bez potrzeby stosowania rozbudowanych interfejsów ani usług zewnętrznych.
CMD LOCAL AI to moja próba stworzenia przejrzystego, natywnego dla terminala sposobu interakcji z modelami AI.

---

## 🚀 Future Ideas

### English
CMD LOCAL AI is a project being developed step by step. Below are the directions I plan to develop in future versions:

- **Support for multiple models simultaneously** - switching between local models without restarting the application.
- **Performance profiling** - real-time viewing of generation time and RAM and CPU usage.
- **Conversation history** - saving and reviewing previous sessions in the form of simple logs.
- **Developer Tools mode** - viewing system prompts, tokens, and raw model responses.
- **Integration with plugins/extensions** - the ability to add custom commands and actions performed by the AI.
- **Support for audio and vision models** - speech and image support for compatible local models.
  
### Polish

CMD LOCAL AI to projekt rozwijany krok po kroku. Poniżej kierunki, które planuję rozwijać w kolejnych wersjach:

- **Wsparcie dla wielu modeli jednocześnie** - przełączanie między lokalnymi modelami bez restartu aplikacji.
- **Profilowanie wydajności** - podgląd czasu generowania oraz zużycia RAM i CPU w czasie rzeczywistym.
- **Historia rozmów** - zapisywanie i przeglądanie poprzednich sesji w formie prostych logów.
- **Tryb Developer Tools** - podgląd promptów systemowych, tokenów i surowych odpowiedzi modelu.
- **Integracja z pluginami / rozszerzeniami** - możliwość dodawania własnych komend i akcji wykonywanych przez AI.
- **Wsparcie dla modeli audio i vision** - obsługa mowy i obrazów dla kompatybilnych modeli lokalnych.


## THANKS FOR READING :)









