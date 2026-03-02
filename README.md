# KI-Maschinenüberwachung

Realistische Simulation einer Produktionsmaschine mit zwei Erkennungssystemen im Direktvergleich:

| | Regelbasiert | KI (Isolation Forest) |
|---|---|---|
| Erkennt Grenzwertverletzungen | ✅ | ✅ |
| Erkennt langsamen Lagerverschleiß | ❌ | ✅ |
| Erkennt neue Sensor-Korrelationen | ❌ | ✅ |
| Erkennt multivariate Muster | ❌ | ✅ |
| Braucht Vorwissen über Anomalien | ✅ Ja | ❌ Nein |

---

## Schnellstart

```bash
# Einmalig: uv installieren (falls nicht vorhanden)
# https://docs.astral.sh/uv/getting-started/installation/

# Abhängigkeiten installieren
uv pip install -r requirements.txt

# Mit Live-Plot (empfohlen)
uv run python main.py

# Nur Terminal-Output
uv run python main.py --no-plot

# Oder zwei Terminals:
# Terminal 1 – Maschine
uv run python generator.py

# Terminal 2 – KI + Regelwerk
uv run python detector.py
```

---

## Demo-Ablauf (`--scenario demo`)

| Schritt | Was passiert | Regelbasiert | KI |
|---------|-------------|-------------|-----|
| 0–150   | Normalbetrieb (Training) | OK | lernt |
| 150–220 | **Lagerverschleiß** – Vibration steigt langsam | ❌ kein Alarm | ✅ erkennt multivariate Abweichung |
| 220–290 | Normalbetrieb (Wartung erledigt) | OK | OK |
| 290–370 | **Thermaldrift** – Temperatur-Sollwert driftet | ❌ unter Schwelle | ✅ erkennt Korrelationsbruch |
| 370–430 | Normalbetrieb | OK | OK |
| 430–530 | **Prozessänderung** – Feuchtigkeit koppelt an Output | ❌ kein Alarm | ✅ erkennt neue Korrelation |
| 530–580 | **Sensordrift** – Temperaturfühler offset | teils | ✅ erkennt systematischen Versatz |

---

## Physikalisches Modell

Die Maschine ist kein Zufallsgenerator – alle Größen sind kausal verknüpft:

```
Solltemperatur ──thermal lag──→ Isttemperatur ──→ Output-Reduktion (Wärmeausdehnung)
Lagerverschleiß ─────────────→ Vibration ────→ Output-Reduktion + Leistungsanstieg
Umgebungsfeuchte (normal) ───→ sehr schwach auf Output
Umgebungsfeuchte (Prozessänderung) → stark auf Output (neues Material)
```

Der Detektor kennt dieses Modell **nicht** – er lernt die Zusammenhänge aus Daten.

---

## Manueller Betrieb

Anomalie-Modus in `anomaly_mode.txt` schreiben (0–4):

```bash
echo 0 > anomaly_mode.txt   # Normal
echo 1 > anomaly_mode.txt   # Lagerverschleiß
echo 2 > anomaly_mode.txt   # Thermaldrift
echo 3 > anomaly_mode.txt   # Prozessänderung
echo 4 > anomaly_mode.txt   # Sensordrift

python generator.py --scenario manual
```

---

## Für echte Maschinen

Die Architektur ist direkt übertragbar:
- `StreamReader` → ersetze durch MQTT / OPC-UA / Modbus reader
- `generator.py` → ersetzt durch echte Sensoranbindung
- Das Feature-Engineering und das Isolation-Forest-Modell bleiben identisch
- Retraining auf neuen Normaldaten: `IsolationForestDetector._train()` erneut aufrufen
