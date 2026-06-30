# AI Message Routing — PoC

Usługa, która **czyta wiadomość pracownika, sama decyduje do którego działu należy, i wysyła ją tam
mailem** — używając lokalnego modelu językowego (LLM) działającego jako **agent**. Agent nie zwraca
„odpowiedzi do sklejenia" w kodzie — **samodzielnie wywołuje narzędzie** `send_email` (mechanizm
*tool / function calling*), zgodnie z wymaganiem zadania.

```
POST /api/v1/route-message  { "email": <nadawca>, "message": <treść> }
        │
        ▼
   Agent (LLM) czyta treść i decyduje
        │
        └─► wywołuje narzędzie  send_email(department = KADRY | HR | HELPDESK | IT | INNE)
                 │
                 └─► narzędzie wysyła maila przez SMTP → MailHog
                     (To: adres działu,  Reply-To: nadawca z requestu)
```

To, że **model sam wywołuje narzędzie** (a nie kod aplikacji po jego decyzji), jest tu celowe i wprost
wymagane — to sedno „AI Agenta" z treści zadania.

## Szybki start

Wymagania: Docker + Docker Compose.

```bash
docker compose up -d --build
```

Podnosi trzy usługi (API czeka, aż Ollama pobierze model — `depends_on: service_healthy`):

| Usługa   | Adres                              | Rola                                       |
|----------|------------------------------------|--------------------------------------------|
| API      | http://localhost:8000/api/v1/docs  | FastAPI + dokumentacja Swagger             |
| MailHog  | http://localhost:8025              | Skrzynka testowa (podgląd wysłanych maili) |
| Ollama   | (wewnętrzny, `ollama:11434`)       | Lokalny LLM, model `Qwen3-4B-Instruct-2507`|

> **Pierwsze uruchomienie trwa dłużej** — Ollama pobiera model (~2,5 GB), a **pierwsze zapytanie**
> rozgrzewa model (na CPU to ~60–90 s, bo model „czyta" raz długą instrukcję systemową). Aplikacja robi
> to automatycznie w tle przy starcie; **każde kolejne zapytanie to już kilka sekund.**

### Szybsze działanie na GPU (opcjonalnie)

Domyślnie stack działa na **CPU** (uruchomi się na każdej maszynie). Jeśli masz kartę NVIDIA, włącz GPU —
model przyspiesza do ~1–3 s na zapytanie:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### Przykład

```bash
curl -X POST http://localhost:8000/api/v1/route-message \
  -H "Content-Type: application/json" \
  -d '{"email":"jan.kowalski@example.com","message":"Chcę iść na urlop w przyszłym miesiącu."}'
```
```json
{"department":"KADRY","target_email":"kadry@firma.pl","sent":true,"detail":"classified and delivered"}
```
Wiadomość pojawia się w MailHogu: **To: kadry@firma.pl**, **Reply-To: jan.kowalski@example.com**.

## Jak działa decyzja o dziale

Pięć działów docelowych. Sednem instrukcji dla modelu jest rozróżnienie **twardych** i **miękkich**
kompetencji, a w razie wątpliwości — bezpieczny wybór `INNE`:

| Dział       | Adres               | Zakres                                                                              |
|-------------|---------------------|-------------------------------------------------------------------------------------|
| `KADRY`     | kadry@firma.pl      | **Twardy HR / płace**: umowy, wynagrodzenia, **urlopy**, L4, PIT/ZUS, świadczenia.  |
| `HR`        | hr@firma.pl         | **Miękki HR**: rekrutacja, onboarding, szkolenia, oceny, konflikty, kultura.        |
| `HELPDESK`  | helpdesk@firma.pl   | **1. linia IT**: reset hasła, dostęp do aplikacji, drobny sprzęt, „jak zrobić X".   |
| `IT`        | it@firma.pl         | **2. linia / infrastruktura**: awarie systemów, sieć/VPN, bezpieczeństwo, integracje.|
| `INNE`      | kontakt@firma.pl    | **Fallback**: cokolwiek niejednoznacznego. Bezpieczny domyślny cel.                 |

## Decyzje architektoniczne

1. **Python + FastAPI + pydantic-ai (slim), bez langchain.** Asynchroniczne API z automatycznym
   Swaggerem i walidacją na Pydanticu. Świadomie minimalny zestaw zależności — do jednego zadania
   klasyfikacji nie potrzeba ciężkiego frameworka agentowego.

2. **Agent wysyła maila przez wywołanie narzędzia (tool / function calling).** Model dostaje narzędzie
   `send_email(department)` i to **on** podejmuje decyzję oraz je wywołuje — wysyłka dzieje się w środku
   narzędzia. Tak wygląda prawdziwy „AI Agent" z treści zadania, a nie klasyfikator z doklejoną w kodzie
   wysyłką. (`app/classifier.py`)

3. **Typ wyliczeniowy (Enum) jako bariera ochronna (guardrail).** Argument narzędzia `department` jest
   typowany na `TargetDepartment` (pięć działów). pydantic-ai **odrzuca każdą wartość spoza listy**, więc
   **model nie jest w stanie wymyślić adresu spoza pięciu działów** — guardrail pilnuje samego wywołania
   narzędzia, nie kruchego parsowania tekstu.

4. **Twardy vs miękki podział kompetencji w instrukcji.** Najtrudniejsze przypadki to KADRY↔HR i
   HELPDESK↔IT. Instrukcja rozdziela je jawnie (formalności płacowe vs ludzie/rozwój; proste prośby vs
   awarie/infrastruktura), z `INNE` jako bezpiecznym domyślnym celem.

   > **Uwaga o nakładających się adresach (świadoma decyzja).** Lista adresów zawierała celowy konflikt:
   > `kadry@` vs `human-resources@` oraz `it@` vs `help-desk@`. Rozważyłem **podział językowy**
   > (PL → kadry, EN → human-resources) — jest kuszący, bo deterministyczny. Odrzuciłem go ze względu na
   > **spójność**: pary `it@`/`help-desk@` nie da się sensownie dzielić językiem (nikt nie routuje zgłoszeń
   > „po polsku → helpdesk"), więc system i tak musi mieć **jedną** filozofię — funkcjonalną. Zastosowanie
   > podziału językowego tylko do HR byłoby wyjątkiem trudniejszym do obrony niż spójna reguła funkcjonalna.

5. **Wybór modelu — podyktowany wymaganiem tool-callingu.** Zadanie wymaga, by agent wysyłał maila przez
   wywołanie narzędzia. To **zawęża** wybór lokalnego modelu — nie każdy mały model to potrafi:
   - `qwen2.5:3b` (pierwszy wybór) w Ollamie **w ogóle nie wywoływał narzędzi** (zweryfikowane: zwracał
     tekst zamiast tool-calla — znany problem z szablonem narzędzi Qwen2.5 w Ollamie).
   - Modele **rozumujące** (np. hybrydowy `qwen3:4b`) wywołują narzędzia, ale generują setki tokenów
     „myślenia" na trywialną decyzję → ~100 s na zapytanie nawet na GPU.
   - Wybrałem **`Qwen3-4B-Instruct-2507`** (wariant *instruct*, **bez trybu rozumowania**): wywołuje
     narzędzie bezpośrednio (kilkadziesiąt tokenów), ma dobrą wielojęzyczność (polski), a wysłany model to
     kwant GGUF od Unsloth (`UD-Q4_K_XL`, ~2,5 GB), pobierany wprost przez Ollamę.

6. **Bezpieczna semantyka błędów (brak „ślepego" 500).** Niedostępny/wolny LLM albo SMTP → `503`
   (usługa chwilowo niedostępna), nigdy blankietowe `500`. Jeśli model mimo wszystko nie wywoła
   narzędzia, też zwracamy `503` zamiast po cichu udawać sukces. Blokujące `smtplib` jest odsunięte do
   puli wątków, więc nie blokuje pętli zdarzeń.

7. **Wykrywanie prompt-injection przed modelem.** Wiadomości próbujące manipulować klasyfikatorem
   („zignoruj instrukcje, odpowiedz X") są wykrywane deterministycznie i kierowane do `INNE` —
   **adversarialny tekst nigdy nie trafia do modelu** (`app/injection.py`).

8. **Konfiguracja ze zmiennych środowiskowych (12-factor).** Adres Ollamy/SMTP, model i timeout są w
   jednej klasie `Settings` (`app/config.py`). Wartości domyślne pasują do `docker-compose` i są
   nadpisywalne przez `.env` / zmienne środowiskowe.

## Testy

```bash
.venv/bin/python -m pytest -q        # 23 testy; offline — bez żywej Ollamy/SMTP
```
Testy podmieniają model na sterowalną atrapę (pydantic-ai `FunctionModel`), która **realnie wywołuje
narzędzie** `send_email` — dzięki temu sprawdzamy całą ścieżkę agenta (wywołanie narzędzia → wysyłka),
a nie tylko fragmenty. Pokrywają m.in.: poprawne wywołanie narzędzia i dostarczenie maila z `Reply-To`,
**guardrail** (argument spoza enuma → brak wysyłki → `503`), sytuację gdy model nie wywoła narzędzia,
krótkie spięcie prompt-injection do `INNE`, awarię LLM/SMTP → `503`, walidację wejścia (`422`), STARTTLS
oraz limit zapytań (`429`).

## Ewaluacja na żywym produkcie

Poza testami jednostkowymi klasyfikator jest mierzony **stress-evalem** (`loop/round-21/eval_run.py`):
74 celowo niechlujne wiadomości (literówki, CAPS, gwara, emoji, mieszanka PL/EN, prompt-injection) × 3 =
**222 wywołania** przez żywy endpoint.

**Wynik (model `Qwen3-4B-Instruct-2507`, CPU):**

| Metryka                                   | Wynik                                            |
|-------------------------------------------|--------------------------------------------------|
| Trafność (run-level, 222 wywołania)       | **97,3%** (216/222)                              |
| Trafność (case-level, większość z 3)      | 97,3% (72/74)                                    |
| Determinizm (wszystkie 3 przebiegi równe) | **100%** (74/74)                                 |
| Niezawodność tool-callingu (błędy / 503)  | **0/222** — model wywołał narzędzie za każdym razem |
| Przecieki prompt-injection                | **0** (18/18 prób odbitych)                      |

Per dział: KADRY / HR / HELPDESK / IT po **100%**; INNE 83% (kategoria-„śmietnik", z natury najtrudniejsza
— 2 graniczne pomyłki typu *vague/offtopic* mylone z HELPDESK/HR, wypisane w `loop/round-21/results.json`).

Najistotniejsze dla wymagania zadania: **0 nieudanych wywołań narzędzia na 222** — mały model *instruct*
korzysta z function-callingu niezawodnie, więc nie jest potrzebne żadne obejście ani fallback. Trafność
jest przy tym **wyższa niż we wcześniejszym wariancie deterministycznym** (93,2% na `qwen2.5:3b`, ten sam
zestaw testowy).

## Struktura

```
app/
  main.py          # FastAPI: endpoint, warmup modelu na starcie, mapowanie błędów na 503
  models.py        # TargetDepartment (Enum) + adresy + modele request/response
  classifier.py    # agent pydantic-ai + narzędzie send_email (tool calling), prompt systemowy
  email_sender.py  # wysyłka SMTP (smtplib): To=dział, Reply-To=nadawca
  injection.py     # deterministyczny detektor prompt-injection (przed modelem)
  config.py        # Settings (env): Ollama/SMTP/model/timeout
  tests/           # pytest (atrapa modelu wołająca narzędzie + mock SMTP)
docker-compose.yml      # api + mailhog + ollama (CPU; healthcheck + kolejność startu)
docker-compose.gpu.yml  # opcjonalny override: Ollama na GPU NVIDIA
Dockerfile              # python:3.11-slim
docker/ollama-entrypoint.sh  # pobiera model przed gotowością kontenera
```

## Znane ograniczenia i uwagi produkcyjne

Świadomie nazwane granice tego PoC i co należałoby zrobić, żeby uciągnął produkcję:

1. **Rozgrzanie modelu / pierwsze zapytanie.** Na CPU pierwsze zapytanie zajmuje ~60–90 s (model raz
   „czyta" długą instrukcję systemową), kolejne ~kilka sekund. Aplikacja rozgrzewa model w tle przy
   starcie. *Produkcyjnie:* trzymać model załadowany (keep-alive) i ewentualnie serwować na GPU.

2. **Przepustowość / współbieżność.** Pojedyncza instancja Ollamy obsługuje żądania sekwencyjnie; przy
   nagłym zalewie maili zapytania ustawiają się w kolejce. *Produkcyjnie:* oddzielić **przyjęcie** maila
   od **klasyfikacji** (kolejka + pula workerów), serwowanie modelu na GPU z batchingiem, autoskalowanie.

3. **Trwałość dostawy.** Gdy klasyfikacja się powiedzie, ale SMTP padnie → `503`, a wiadomość przepada
   (brak dead-letter/retry/persystencji). *Produkcyjnie:* kolejka wychodząca z ponawianiem + dead-letter.

4. **Walidacja i nadużycia.** Wejście ma limit długości (`max_length=4000` → `422`) chroniący przed
   zalaniem kontekstu modelu; endpoint ma rate-limiting (slowapi, domyślnie `300/min`). Brak
   **uwierzytelniania** endpointu — poza zakresem PoC.

5. **Zakres językowy.** Instrukcja jest polskocentryczna (obsługuje PL + wtrącenia EN); inne języki są
   nieprzewidziane — świadomy zakres dla polskiej firmy.

6. **Dostawa testowa.** MailHog to skrzynka testowa — nic nie wychodzi na zewnątrz. Produkcyjnie wystarczy
   podmienić `SMTP_*` (z opcjonalnym STARTTLS/login, już wspieranym przez config).
