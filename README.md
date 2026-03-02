# KI-Maschinenüberwachung mit LLM-Advisor

Realistische Simulation einer industriellen Produktionsmaschine – mit drei Schichten
der Anomalieerkennung im Direktvergleich:

```
Sensordaten  →  Regelbasierter Detektor   →  Alarm bei Grenzwertverletzung
             →  KI (Isolation Forest)     →  Alarm bei multivariaten Mustern
             →  LLM-Advisor (OpenRouter)  →  Diagnose + Handlungsempfehlungen
```

---

## Was hier passiert

### 1. Maschinensimulator (`generator.py`)

Fünf physikalisch gekoppelte Sensoren werden in Echtzeit generiert:

| Sensor | Einheit | Modell |
|---|---|---|
| `temperatur` | °C | First-Order-Thermalmodell mit Produktionszyklus + Umgebungsdrift |
| `vibration` | mm/s | Mechanisches Modell – steigt mit Lagerverschleiß |
| `humidity` | % | Umgebungsmodell – antikorrelliert zur Temperatur |
| `output` | Stück/min | Abhängig von Temp, Vibration, Feuchte (modenabhängig) |
| `power_kw` | kW | Gekoppelt an Output, Temperaturlast und Reibungsverluste |

Der Generator läuft nach einem festen **Demo-Schedule** mit 8 Anomalie-Phasen:

| Schritte | Anomalie | Regelwerk | KI |
|---|---|---|---|
| 0 – 370 | Normalbetrieb (Training) | OK | lernt |
| 370 – 460 | **Lagerverschleiß I** – Vibration steigt langsam | ❌ | ✅ |
| 570 – 640 | **Thermaldrift I** – Temperatur-Sollwert driftet | ❌ | ✅ |
| 750 – 820 | **Überhitzung** – harter Temperatursprung | ✅ | ✅ |
| 930 – 1050 | **Prozessänderung I** – Feuchtigkeit koppelt an Output | ❌ | ✅ |
| 1160 – 1240 | **Lagerverschleiß II** – zweite Episode | ❌ | ✅ |
| 1340 – 1420 | **Sensordrift** – Temperaturfühler verstimmt sich | ❌ | ✅ |
| 1530 – 1610 | **Thermaldrift II** | ❌ | ✅ |
| 1700 – 1820 | **Prozessänderung II** | ❌ | ✅ |

Zwischen den Phasen: 110-Schritte-Normalpausen (= 33 s real) → mehrere Episoden
passen in das 3-Minuten-Gedächtnisfenster des Advisors.

---

### 2. Anomalie-Detektor (`detector.py`)

#### Regelbasierter Detektor
Feste Min/Max-Grenzen pro Sensor. Schlägt an, wenn ein einzelner Kanal seinen
Grenzwert überschreitet. Einfach, transparent, aber blind für langsame oder
multivariate Muster.

#### KI-Detektor (Isolation Forest)
Lernt in der Trainingsphase (Schritte 0–300) die **normale** Verteilung aus
**70 Features**, die aus den Rohsensoren berechnet werden:

- Rolling Z-Scores (kurz + lang) – zyklusnormiert
- Änderungsraten (roc1, roc5, roc10, roc20, roc30) – nur steigende Flanken
- Rolling Std-Verhältnisse – erkennt Varianzänderungen
- Kreuzterme (z.B. `z_vib × z_power`) – multivariate Muster
- Rolling Korrelationen (z.B. `corr_hum_out`) – erkennt neue Kopplungen
- Effizienzproxy `power / output` – Lagerverschleiß-Signatur

Der Detektor hat **keinen Zugriff** auf die Anomalie-Labels oder den Schedule.
Er entscheidet rein aus statistischer Abweichung vom gelernten Normalzustand.

---

### 3. Symptom-Tracking (`advisor.py`)

Bei jedem Alarm-Schritt speichert der **SymptomTracker** ein Ereignis mit:
- Zeitstempel
- KI-Erklärungstext (welche Features sind auffällig)
- Symptom-Tags (automatisch aus dem Erklärungstext geparst)
- Regelverstoße (falls vorhanden)
- Sensorwerte zum Zeitpunkt des Alarms

Das Fenster umfasst die **letzten 600 Schritte ≈ 3 Minuten Maschinenzeit**.
Aufeinanderfolgende Alarm-Schritte werden zu Episoden zusammengefasst.

Bei einem neuen Alarm wird der vollständige Verlauf ausgegeben:

```
┌── 📋 SYMPTOM-VERLAUF (Schritte 370–930, ≈ 2.8 min) ──────────┐
│  [370–450] 🟡 Nur KI   Vibration erhöht (+3.1σ vom gleit.…   │
│  [570–632] 🟡 Nur KI   Temperatur steigt systematisch (+4σ)… │
│  [750–815] 🔴 KI+Regel Temperatur erhöht | Output gesunken…  │
└───────────────────────────────────────────────────────────────┘
```

---

### 4. Wissensdatenbank (`knowledge_db.py`)

Eine erdachte Industrie-Diagnosedatenbank mit **19 Einträgen**, die
Symptomkombinationen auf Ursachen und Maßnahmen abbildet:

```
Symptom-Tags (aus KI-Erklärung geparst)
       ↓
  lookup(tags)
       ↓
Passende KB-Einträge (sortiert nach Dringlichkeit)
```

Dringlichkeitsstufen: 🚨 kritisch → ⚠️ hoch → 🔶 mittel → 🔵 niedrig

Beispiel-Einträge:

| ID | Name | Auslöser |
|---|---|---|
| KB-001 | Lagerverschleiß – Früherkennung | `vib_erhöht` |
| KB-002 | Lagerverschleiß – Fortgeschritten | `vib_erhöht` + `vib_trend` |
| KB-020 | Akute Überhitzung – Notfall | `temp_erhöht` + `output_gesunken` |
| KB-030 | Unerwartete Materialreaktion | `hum_out_kopplung` |
| KB-040 | Temperatursensor-Drift | `temp_trend` ohne sonstige Symptome |
| KB-060 | Überhitztes Lager | `vib_erhöht` + `temp_erhöht` |
| KB-080 | Wiederkehrende Anomalien | alle schweren Symptome |

---

### 5. LLM-Advisor (`advisor.py`)

Bei jedem neuen Alarm wird ein **LLM via OpenRouter** asynchron befragt
(blockiert den Detektor-Loop nicht). Der Prompt enthält:

- Aktuelle Sensorwerte
- KI-Diagnosetext + Regelverstoße
- Formatierten Symptom-Verlauf der letzten 3 Minuten
- Bis zu 4 passende Wissensdatenbank-Einträge

Das LLM antwortet strukturiert auf Deutsch:
1. Wahrscheinliche Ursache
2. 3 Sofortmaßnahmen
3. Dringlichkeit
4. Folgerisiko bei Untätigkeit

Ausgabe im Terminal:
```
┌── 🤖 KI-BERATER (qwen3-30b-a3b) [Schritt 930] ───────────────┐
│  1. URSACHE: Die Kombination aus Feuchtigkeits-Output-        │
│  Kopplung und dem vorherigen Lagermuster deutet auf …         │
│  2. SOFORTMASSNAHMEN:                                         │
│    1. Materialcharge prüfen – Feuchtegehalt messen …          │
│  3. DRINGLICHKEIT: innerhalb 1 Stunde – weil …               │
│  4. FOLGERISIKO: Weiterproduktion gefährdet die Charge …      │
└───────────────────────────────────────────────────────────────┘
```

---

## Warum KI dem Regelwerk überlegen ist

Ein klassisches regelbasiertes System prüft jeden Sensor einzeln gegen feste Grenzwerte –
zum Beispiel: *„Alarm, wenn Vibration > 1,60 mm/s"*. Das funktioniert zuverlässig für
offensichtliche Ausfälle, hat aber einen strukturellen Nachteil: Es sieht immer nur
einen Kanal zur Zeit und weiß nichts über das Zusammenspiel der Sensoren.

Industrielle Anomalien entstehen aber selten aus dem Nichts. Sie entwickeln sich langsam
und zeigen sich zuerst als subtile Verschiebung im **Verhältnis** mehrerer Größen zueinander –
lange bevor ein einzelner Wert seinen Grenzwert reißt.

### KI-Stufe 1: Isolation Forest (scikit-learn + pandas)

Der Isolation Forest lernt in einer Trainingsphase (300 Messwerte bei Normalbetrieb)
wie sich die Maschine **typischerweise** verhält – nicht als einzelne Grenzwerte, sondern
als hochdimensionale Normalverteilung über 70 berechnete Features.

Was er dadurch erkennt, was das Regelwerk nicht kann:

**Lagerverschleiß** – Die Vibration steigt über 70 Schritte um 0,54 mm/s an.
Das Regelwerk hat eine Grenze bei 1,60 mm/s – die wird nie erreicht.
Der Isolation Forest erkennt: Vibration *steigt systematisch*, gleichzeitig
steigt der Energiebedarf pro produziertem Stück (`power / output`) und die
Vibrations-Leistungs-Korrelation verändert sich. Kein Einzelsensor schlägt an –
das multivariate Muster tut es.

**Thermaldrift** – Die Temperatur driftet langsam 8 °C nach oben (Grenzwert: 82 °C,
Istwert bleibt bei ~73 °C). Das Regelwerk schläft. Der Isolation Forest sieht den
anhaltenden Aufwärtstrend im 30-Schritte-ROC und den beginnenden Output-Rückgang
durch Wärmeausdehnung – und schlägt an.

**Prozessänderung** – Alle Sensorwerte bleiben im Normalbereich. Die Grenzwerte
sind nie in Gefahr. Aber: Die rollende Korrelation zwischen Luftfeuchtigkeit und
Output (`corr_hum_out`) springt von nahezu null auf stark negativ – weil ein neues
Material eingesetzt wurde, das feuchteempfindlich ist. Das Regelwerk hat keine
Chance, das zu sehen. Der Isolation Forest erkennt die veränderte Struktur sofort.

**Sensordrift** – Der Temperatursensor verstimmt sich um 4 °C. Der angezeigte Wert
liegt bei ~69 °C, der echte Istwert bei ~65 °C. Kein Grenzwert verletzt.
Der Isolation Forest erkennt den systematischen Temperaturtrend, während Leistung,
Vibration und Output stabil bleiben – ein Muster, das fast ausschließlich
auf einen Kalibrierungsfehler hindeutet.

```
Regelwerk:  prüft 5 Sensoren × 2 Grenzwerte = 10 Bedingungen
KI:         lernt eine 70-dimensionale Normalverteilung –
            erkennt Abweichungen im Raum der Beziehungen zwischen Sensoren
```

Das ist der Kernunterschied: Das Regelwerk reagiert auf **Zustände**.
Die KI reagiert auf **Veränderungen im Verhalten** – auch dann, wenn alle
Einzelwerte noch im grünen Bereich liegen.

---

### KI-Stufe 2: LLM-Advisor (Sprachmodell via OpenRouter)

Der Isolation Forest meldet *dass* etwas nicht stimmt und *welche Features* auffällig sind.
Er kann aber nicht sagen *warum* das wahrscheinlich passiert oder *was jetzt zu tun ist*.

Hier kommt das Sprachmodell ins Spiel – und es löst ein anderes Problem als der
Isolation Forest: Es überbrückt die Lücke zwischen Messwert und Handlung.

Ein Wartungstechniker, der einen Alarm bekommt, braucht keine Sigma-Werte.
Er braucht: *Was ist kaputt? Was soll ich jetzt tun? Wie dringend ist das?*

Das LLM bekommt drei Dinge:

1. **Den aktuellen Kontext** – Sensorwerte, KI-Diagnosetext, Regelverstoße
2. **Den Verlauf der letzten 3 Minuten** – eine Timeline aller vorherigen
   Alarm-Episoden aus dem SymptomTracker. So sieht das Modell nicht nur den
   aktuellen Alarm, sondern auch, ob davor schon Vibrations- oder Temperaturprobleme
   aufgefallen sind.
3. **Vorgefiltertes Fachwissen** – aus der Wissensdatenbank werden automatisch die
   passenden Einträge für die erkannten Symptomkombinationen ausgewählt und mitgegeben.

Das Ergebnis ist kein generisches *„bitte einen Techniker rufen"*, sondern eine
kontextbewusste Einschätzung: Das Modell sieht, dass in den letzten 3 Minuten
bereits ein Lager- und ein Temperaturproblem aufgetreten sind, und kann daraus
schließen, dass es sich wahrscheinlich um ein überhitztes Lager handelt – eine
Diagnose, die weder der Isolation Forest noch das Regelwerk allein stellen könnten.

```
Regelwerk:   „Grenzwert überschritten"
Isolation Forest:  „Multivariates Muster – Vibration + Effizienz auffällig"
LLM-Advisor: „Wahrscheinlich überhitztes Lager. Sofortmaßnahmen:
              1. IR-Thermometer an Lagerstelle, 2. Notschmierung,
              3. Produktionstakt auf 70% reduzieren.
              Dringlichkeit: sofort – Lagerausfall binnen 2h möglich."
```

Die drei Schichten ergänzen sich: Das Regelwerk fängt das Offensichtliche.
Der Isolation Forest erkennt das Subtile. Das LLM macht daraus eine Handlung.

---

## Schnellstart

### Voraussetzungen

- Python 3.11+ (empfohlen: via [uv](https://docs.astral.sh/uv/))
- OpenRouter API-Key → [openrouter.ai](https://openrouter.ai)

### Installation

```bash
# Abhängigkeiten installieren
uv pip install -r requirements.txt

# API-Key hinterlegen
cp .env.example .env
# .env öffnen und OPENROUTER_API_KEY eintragen
```

### Starten

```bash
# Mit Live-Plot (4 Panels: Sensoren, KI-Score, Regelwerk, Vergleich)
uv run python main.py

# Nur Terminal (empfohlen wenn kein Display vorhanden)
uv run python main.py --no-plot

# Schnelleres Tempo
uv run python main.py --interval 0.1

# Oder getrennt in zwei Terminals:
uv run python generator.py   # Terminal 1 – Maschine
uv run python detector.py    # Terminal 2 – KI + Berater
```

---

## Dateien

```
kisimulation/
├── generator.py      Maschinensimulator – schreibt stream.csv
├── detector.py       Isolation Forest + Regelwerk + Symptom-Tracking
├── knowledge_db.py   Industrie-Diagnosedatenbank (19 Einträge)
├── advisor.py        SymptomTracker + LLM-Advisor (OpenRouter)
├── main.py           Orchestrator + Live-Plot (matplotlib)
├── requirements.txt
├── .env              API-Key (nicht im Repo, in .gitignore)
└── .env.example      Vorlage für .env
```

---

## Physikalisches Modell

Alle Sensoren sind kausal verknüpft – kein Zufallsgenerator:

```
Solltemperatur + Anomalie-Offset
        │ thermal lag (τ ≈ 20 Schritte)
        ▼
    Isttemperatur ──────────────────────────→ Output-Reduktion (Wärmeausdehnung)
        │                                     Power ∝ Temp
        ▼
    Humidity ← antikorrelliert (normal)
        │
        └──→ Output-Reduktion (schwach normal / stark bei Prozessänderung)

Lagerverschleiß → Vibration ────────────────→ Output-Reduktion
                       │                      Power ∝ Vibration²
                       └──→ vib_pwr_kopplung (Rolling Korrelation)

Sensoroffset → Temperaturfühler liest zu hoch (Istwert normal)
```

Der Detektor kennt dieses Modell **nicht** – er lernt es aus den Daten.

---

## Manueller Betrieb

Anomalie-Modus live wechseln:

```bash
echo 0 > anomaly_mode.txt   # Normal
echo 1 > anomaly_mode.txt   # Lagerverschleiß
echo 2 > anomaly_mode.txt   # Thermaldrift
echo 3 > anomaly_mode.txt   # Prozessänderung
echo 4 > anomaly_mode.txt   # Sensordrift
echo 5 > anomaly_mode.txt   # Überhitzung (Notfall)

uv run python generator.py --scenario manual
```

---

## Auf echte Maschinen übertragen

Die Architektur ist direkt produktionsfähig:

| Komponente | Demo | Produktion |
|---|---|---|
| Datenquelle | `generator.py` → CSV | MQTT / OPC-UA / Modbus → `StreamReader` ersetzen |
| Feature-Engineering | `compute_features()` | identisch übernehmen |
| Modell | `IsolationForestDetector` | identisch – Retraining auf Produktionsdaten |
| Wissensdatenbank | `knowledge_db.py` | eigene Diagnoseregeln ergänzen |
| LLM-Advisor | OpenRouter / qwen | beliebiges OpenAI-kompatibles Modell |
