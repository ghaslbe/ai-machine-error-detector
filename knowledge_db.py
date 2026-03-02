#!/usr/bin/env python3
"""
Erdachte Diagnose-Wissensdatenbank für Industriemaschinen.

Bildet Symptomkombinationen (aus der KI-Erklärung) auf Ursachen und
praxisnahe Handlungsempfehlungen ab.

Struktur:
  - SYMPTOM_PATTERNS   : keyword→tag Mapping für den _explain()-Output
  - SYMPTOM_LABELS     : menschenlesbare Namen der Tags
  - KNOWLEDGE_BASE     : Liste von Diagnose-Einträgen mit match_any/match_all
  - match_symptoms()   : Text → [tags]
  - lookup()           : [tags] → passende KB-Einträge (sortiert nach Dringlichkeit)
"""

import re
from typing import Any


# ── Symptom-Klassifikation ──────────────────────────────────────────────────────
# Mappe Schlüsselwörter aus dem _explain()-Output auf strukturierte Tags.

SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "vib_erhöht":        [r"Vibration erh.ht", r"Vibration.+steigt.+\d+σ"],
    "vib_trend":         [r"Vibration steigt systematisch"],
    "vib_gesunken":      [r"Vibration gesunken", r"Vibration.+fällt"],
    "temp_erhöht":       [r"Temperatur erh.ht", r"Temperatur.+steigt.+\d+σ"],
    "temp_trend":        [r"Temperatur steigt systematisch"],
    "temp_gesunken":     [r"Temperatur gesunken", r"Temperatur.+fällt"],
    "output_gesunken":   [r"Output.+gesunken", r"Output.+fällt"],
    "output_trend":      [r"Output.+steigt systematisch"],
    "power_erhöht":      [r"Leistungsaufnahme erh.ht", r"Leistungsaufnahme.+steigt"],
    "power_gesunken":    [r"Leistungsaufnahme gesunken"],
    "effizienz_sink":    [r"Leistung/Stück erh.ht"],
    "effizienz_gut":     [r"Leistung/Stück gesunken"],
    "hum_out_kopplung":  [r"Feuchte.Output-Kopplung"],
    "vib_pwr_kopplung":  [r"Vibration.Leistung-Kopplung"],
    "temp_out_kopplung": [r"Temp.Output-Kopplung"],
    "humidity_erhöht":   [r"Feuchtigkeit erh.ht"],
    "humidity_gesunken": [r"Feuchtigkeit gesunken"],
    "multivariat":       [r"Multivariates Muster"],
}

SYMPTOM_LABELS: dict[str, str] = {
    "vib_erhöht":        "Vibration erhöht",
    "vib_trend":         "Vibration steigt systematisch",
    "vib_gesunken":      "Vibration gesunken",
    "temp_erhöht":       "Temperatur erhöht",
    "temp_trend":        "Temperatur steigt systematisch",
    "temp_gesunken":     "Temperatur gesunken",
    "output_gesunken":   "Output (Stückzahl) gesunken",
    "output_trend":      "Output steigt systematisch",
    "power_erhöht":      "Leistungsaufnahme erhöht",
    "power_gesunken":    "Leistungsaufnahme gesunken",
    "effizienz_sink":    "Energie-Effizienz gesunken (mehr Strom/Stück)",
    "effizienz_gut":     "Energie-Effizienz verbessert",
    "hum_out_kopplung":  "Neue Feuchtigkeits-Output-Kopplung erkannt",
    "vib_pwr_kopplung":  "Veränderte Vibrations-Leistungs-Kopplung",
    "temp_out_kopplung": "Veränderte Temperatur-Output-Kopplung",
    "humidity_erhöht":   "Luftfeuchtigkeit erhöht",
    "humidity_gesunken": "Luftfeuchtigkeit gesunken",
    "multivariat":       "Multivariates Muster (kein Einzelsensor dominant)",
}


def match_symptoms(explanation: str) -> list[str]:
    """Parst den _explain()-Text → Liste von Symptom-Tags."""
    if not explanation:
        return []
    tags = []
    for tag, patterns in SYMPTOM_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, explanation, re.IGNORECASE):
                tags.append(tag)
                break
    return tags


# ── Wissensdatenbank ────────────────────────────────────────────────────────────
# Jeder Eintrag:
#   id          – eindeutige Kennung
#   name        – Kurzbezeichnung
#   match_any   – mind. EINES dieser Tags muss vorhanden sein
#   match_all   – ALLE diese Tags müssen vorhanden sein
#   exclude     – KEINES dieser Tags darf vorhanden sein
#   urgency     – kritisch | hoch | mittel | niedrig
#   timeframe   – Handlungsfenster
#   cause       – wahrscheinliche Ursache (Freitext)
#   actions     – konkrete Maßnahmen
#   monitoring  – Empfehlung für Folgeüberwachung

KNOWLEDGE_BASE: list[dict[str, Any]] = [

    # ── Lagerverschleiß (Bearing Wear) ─────────────────────────────────────

    {
        "id": "KB-001",
        "name": "Lagerverschleiß – Früherkennung",
        "match_any": ["vib_erhöht"],
        "match_all": [],
        "exclude": [],
        "urgency": "mittel",
        "timeframe": "innerhalb 8 Stunden",
        "cause": (
            "Erhöhte Vibration deutet auf beginnenden Lagerverschleiß hin. "
            "Der Schmiermittelfilm bricht lokal durch, Reibung in den Wälzkörpern steigt. "
            "Typisch nach 70–80 % der Lager-Nennlebensdauer oder bei Überlastung."
        ),
        "actions": [
            "Lagerspiele an Hauptwelle und Antriebseinheit messen (Grenzwert: < 0,15 mm)",
            "Schmierstoffprobe entnehmen – Partikelzählung und Metallspäne prüfen",
            "Thermografie der Lagerstellen durchführen (Grenzwert: < 75 °C)",
            "Ersatzlager (SKF 6205-2RS o. äquivalent) aus Lager vorbereiten",
            "Wartungsintervall vorübergehend auf 72 h verkürzen",
        ],
        "monitoring": "Vibration alle 15 min auf Trend prüfen – Grenzwert 0,9 mm/s",
    },

    {
        "id": "KB-002",
        "name": "Lagerverschleiß – Fortgeschrittene Phase",
        "match_any": [],
        "match_all": ["vib_erhöht", "vib_trend"],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "innerhalb 2 Stunden",
        "cause": (
            "Systematisch steigender Vibrationstrend kombiniert mit erhöhtem Absolutniveau. "
            "Die Wälzkörper beginnen, die Laufbahn zu beschädigen (Pittingbildung). "
            "Ohne Eingriff droht Lagerausfall innerhalb weniger Stunden."
        ),
        "actions": [
            "Produktionsgeschwindigkeit auf 70 % reduzieren",
            "Stoßimpulsmessung (SPM/HFD) an kritischen Lagern sofort durchführen",
            "Schmierstoffzufuhr verdoppeln – Nachschmierung alle 30 min",
            "Lagerwechsel für Ende der aktuellen Schicht einplanen",
            "Ersatzteilversorgung im ERP prüfen, ggf. Expressbestellung auslösen",
        ],
        "monitoring": "Kontinu. Monitoring – bei Vibration > 1,2 mm/s sofortige Notabschaltung",
    },

    {
        "id": "KB-003",
        "name": "Lagerverschleiß mit Effizienzeinbruch",
        "match_any": [],
        "match_all": ["vib_erhöht", "effizienz_sink"],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "sofort prüfen",
        "cause": (
            "Erhöhte Reibungsverluste im Lager treiben den Energiebedarf pro Stück hoch. "
            "Kombination aus mechanischem Verlust und Output-Einbruch ist typisch "
            "für die Endphase vor einem Lagerausfall."
        ),
        "actions": [
            "Antriebsstrom des Hauptmotors messen (Leerlauf vs. Volllast vergleichen)",
            "Anlaufmoment an Getriebewelle prüfen – erhöhtes Schleppmoment identifiziert Lagerproblem",
            "Maschinenbeobachtung durch qualifizierten Instandhalter einleiten",
            "Laufende Produktion auf geplanten Wartungsstopp überschreiben",
            "Unfallgefahr durch Lagerausfall bewerten – ggf. Schutzzone einrichten",
        ],
        "monitoring": "Strom und Vibration jede 5 min; bei weiterer Verschlechterung Stopp",
    },

    {
        "id": "KB-004",
        "name": "Wiederkehrender Lagerverschleiß – systemische Ursache",
        "match_any": ["vib_erhöht", "vib_trend"],
        "match_all": [],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "vor nächster Schicht klären",
        "cause": (
            "Lager-Symptome traten bereits in einer früheren Episode auf. "
            "Einfacher Lagertausch beseitigt möglicherweise nicht die Grundursache "
            "(falsche Lagergröße, Überlast, Schmierfehler, Ausrichtungsfehler)."
        ),
        "actions": [
            "Lagertyp und -größe gegen Spezifikation prüfen",
            "Wellenausrichtung kontrollieren (max. 0,05 mm Fluchtungsfehler)",
            "Schmierstoffqualität und -menge gegen Herstellervorgabe prüfen",
            "Belastungsprofil der letzten 48 h auswerten – Überlastereignisse identifizieren",
            "Root-Cause-Analyse (5-Why) für wiederkehrende Vibrationsanomalien einleiten",
        ],
        "monitoring": "Vibration täglich dokumentieren, Trend über 7 Tage auswerten",
    },

    # ── Thermaldrift ────────────────────────────────────────────────────────

    {
        "id": "KB-010",
        "name": "Thermaldrift – Kühlsystemverdacht",
        "match_any": ["temp_erhöht", "temp_trend"],
        "match_all": [],
        "exclude": ["vib_erhöht", "power_erhöht"],
        "urgency": "mittel",
        "timeframe": "innerhalb 4 Stunden",
        "cause": (
            "Langsam steigender Temperaturtrend ohne Vibrations- oder Leistungsauffälligkeit. "
            "Typische Ursachen: Verschmutzter Wärmetauscher, reduzierter Kühlmittelfluss, "
            "blockierte Belüftung oder erhöhte Umgebungstemperatur."
        ),
        "actions": [
            "Kühlmittelstand im Ausgleichsbehälter prüfen",
            "Wärmetauscher auf Verschmutzung und Kalkablagerungen inspizieren",
            "Kühlmittelpumpe auf Fördermenge prüfen (Sollwert: ≥ 12 l/min)",
            "Lüftungsgitter und Filtereinsätze reinigen",
            "Umgebungstemperatur im Maschinenraum messen (Grenzwert: 35 °C)",
        ],
        "monitoring": "Temperaturtrend alle 30 min bewerten",
    },

    {
        "id": "KB-011",
        "name": "Thermaldrift mit Output-Verlust",
        "match_any": [],
        "match_all": ["temp_erhöht", "output_gesunken"],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "innerhalb 1 Stunde",
        "cause": (
            "Thermische Ausdehnung bei erhöhter Betriebstemperatur verändert Maße in "
            "Werkzeugen und Führungen – Prozessqualität und Taktzahl sinken messbar. "
            "Kühlsystem ist überlastet oder teilausgefallen."
        ),
        "actions": [
            "Werkzeugtemperatur direkt messen – auf thermische Ausdehnung prüfen",
            "Kühlung auf maximale Leistung stellen",
            "Produktionstakt vorübergehend auf 85 % reduzieren",
            "Qualitätsprüfung Fertigteile intensivieren (Maßhaltigkeit, Oberfläche)",
            "Notfall-Lüftungsöffnungen (falls vorhanden) aktivieren",
            "Ursachenanalyse Kühlsystem: Pumpe, Thermostat, Leitungen",
        ],
        "monitoring": "Temperatur und Output-Rate jede Minute überwachen",
    },

    {
        "id": "KB-012",
        "name": "Temperatur-Output-Kopplung verändert",
        "match_any": ["temp_out_kopplung"],
        "match_all": [],
        "exclude": [],
        "urgency": "mittel",
        "timeframe": "innerhalb 4 Stunden",
        "cause": (
            "Die statistische Beziehung zwischen Temperatur und Output hat sich verändert. "
            "Mögliche Ursachen: geänderter Werkstoff (andere Wärmeausdehnung), "
            "neue Bearbeitungsparameter oder Änderung am Kühlsystem."
        ),
        "actions": [
            "Prozessparameter gegen Freigabeprotokoll vergleichen",
            "Materialcharge auf Änderungen prüfen",
            "Letzte Einstellungsänderungen am System dokumentieren und ggf. rückgängig machen",
            "Fertigungsqualität intensiv prüfen",
        ],
        "monitoring": "Temp-Output-Verhältnis stündlich loggen",
    },

    # ── Überhitzung (Overheat) ──────────────────────────────────────────────

    {
        "id": "KB-020",
        "name": "Akute Überhitzung – Notfall",
        "match_any": [],
        "match_all": ["temp_erhöht", "output_gesunken"],
        "exclude": [],
        "urgency": "kritisch",
        "timeframe": "sofort",
        "cause": (
            "Kritische Temperaturüberschreitung mit gleichzeitigem Output-Einbruch. "
            "Mögliche Ursachen: Kühlmittelversagen, Motorblockade, Materialstau "
            "oder Kurzschluss in der Heizung. Brandgefahr!"
        ),
        "actions": [
            "⚡ SOFORTSTOPP – Maschine über Not-Aus sichern",
            "Kühlmittelversorgung sofort prüfen – Pumpenausfall identifizieren",
            "Motortemperatur und Wicklungsschutz prüfen",
            "Antriebsstrang auf Blockierungen untersuchen",
            "Maschinenumgebung auf Brandentwicklung kontrollieren",
            "Erst nach vollständiger Abkühlung (< 50 °C) Neustart prüfen",
            "Qualitäts-Rückhalte für letzten Produktionsblock einleiten",
        ],
        "monitoring": "Kontinuierliche Brandüberwachung – Feuerlöscher bereithalten",
    },

    {
        "id": "KB-021",
        "name": "Überhitzung mit Leistungsspitze",
        "match_any": [],
        "match_all": ["temp_erhöht", "power_erhöht"],
        "exclude": [],
        "urgency": "kritisch",
        "timeframe": "sofort",
        "cause": (
            "Gleichzeitige Temperatur- und Leistungserhöhung. Elektrischer oder "
            "mechanischer Kurzschluss möglich; Motor läuft im Überlastbereich. "
            "Wicklungsschäden und Bauteilversagen wahrscheinlich."
        ),
        "actions": [
            "⚡ SOFORTSTOPP – Leistungsschalter öffnen",
            "Motorschutzschalter auf Auslösung prüfen",
            "Antrieb auf Blockierung, Kurzschluss oder Phasenfehler untersuchen",
            "Elektrische Anlage durch E-Fachkraft freigeben lassen vor Neustart",
        ],
        "monitoring": "Keine Wiederinbetriebnahme ohne E-Fachkraft-Freigabe",
    },

    # ── Prozessänderung (Material-/Rezeptwechsel) ───────────────────────────

    {
        "id": "KB-030",
        "name": "Unerwartete Materialreaktion – Feuchtigkeitskopplung",
        "match_any": ["hum_out_kopplung"],
        "match_all": [],
        "exclude": [],
        "urgency": "mittel",
        "timeframe": "innerhalb 2 Stunden",
        "cause": (
            "Eine neu aufgetretene Feuchtigkeits-Output-Kopplung deutet auf einen "
            "Materialwechsel oder veränderte Rohstoffcharge hin. Hygroskopische Materialien "
            "(Kunststoffe, Granulate, Fasern) reagieren bei Feuchteänderungen mit "
            "Viskositäts- oder Geometrieänderungen, die den Fertigungsprozess stören."
        ),
        "actions": [
            "Materialcharge und Lieferschein der aktuellen Charge prüfen",
            "Feuchtegehalt des Rohmaterials messen (Grenzwert: < 0,2 %)",
            "Rezeptparameter auf Feuchtekorrektur prüfen (Trocknungszeit, Zylindertemperatur)",
            "Vortrocknung für 2 h bei 80 °C einplanen",
            "Qualitätsprüfung der aktuellen Produktionsmenge (Maßhaltigkeit, Gewicht)",
            "Rückmeldung an Einkauf/Lieferant über Chargenveränderung",
        ],
        "monitoring": "Feuchtigkeits- und Output-Korrelation stündlich prüfen",
    },

    {
        "id": "KB-031",
        "name": "Prozessinstabilität – aktiver Feuchteeinfluss",
        "match_any": [],
        "match_all": ["hum_out_kopplung", "output_gesunken"],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "sofort prüfen",
        "cause": (
            "Aktive Feuchtigkeits-Output-Kopplung mit messbarem Output-Einbruch. "
            "Materialquell- oder Viskositätseffekte beeinflussen den Prozess direkt. "
            "Weiterproduktion gefährdet Qualität der gesamten Charge."
        ),
        "actions": [
            "Produktion unterbrechen – Materialprobe ziehen und Feuchte analysieren",
            "Alternative Materialcharge aus Lager bereitstellen und Probe fahren",
            "Maschinenbediener über Prozessinstabilität informieren",
            "100 %-Endkontrolle der letzten 30-min-Produktion anordnen",
        ],
        "monitoring": "Output jede 5 min; bei weiterer Verschlechterung Produktionsstopp",
    },

    # ── Sensordrift ─────────────────────────────────────────────────────────

    {
        "id": "KB-040",
        "name": "Temperatursensor-Drift",
        "match_any": ["temp_trend", "temp_erhöht"],
        "match_all": [],
        "exclude": ["output_gesunken", "power_erhöht", "vib_erhöht"],
        "urgency": "niedrig",
        "timeframe": "innerhalb 24 Stunden",
        "cause": (
            "Systematischer Temperaturtrend ohne andere Auffälligkeiten "
            "(Leistung, Output, Vibration normal) ist typisch für Kalibrierungsdrift "
            "des Temperatursensors. PT100 oder Thermoelement kann sich durch Alterung, "
            "Vibration oder Feuchtigkeitseintritt verstimmen."
        ),
        "actions": [
            "Referenztemperaturmessung mit kalibriertem Handmessgerät an der Messstelle",
            "Sensorkabel auf Leitungswiderstand und Feuchteeintritt prüfen",
            "Sensor-Offset im Prozessleitsystem dokumentieren",
            "Kalibrierschein des Sensors prüfen – ggf. Neukalibrierung beauftragen",
            "Bei Abweichung > 3 °C: Sensor tauschen",
        ],
        "monitoring": "Täglicher Vergleich Handsonde vs. Prozesssensor für 7 Tage",
    },

    {
        "id": "KB-041",
        "name": "Multivariates Muster – Kalibrierungsverdacht",
        "match_any": ["multivariat"],
        "match_all": [],
        "exclude": [],
        "urgency": "niedrig",
        "timeframe": "innerhalb 48 Stunden",
        "cause": (
            "Multivariates Muster ohne dominanten Einzelsensor – KI erkennt statistische "
            "Abweichungen in mehreren Kanälen gleichzeitig. Häufige Ursachen: "
            "Messkettenfehler, Netzstörung, EMV-Einstreuung oder gleichzeitiger Drift "
            "mehrerer Sensoren."
        ),
        "actions": [
            "Vollständige Sensorkalibrierung aller 5 Messkanäle planen",
            "Erdung und Schirmung der Signalleitungen prüfen",
            "Spannungsversorgung Sensorik auf Rippelspannung messen",
            "Betriebszustand mit Referenzmessung aus letzter Wartung vergleichen",
        ],
        "monitoring": "KI-Anomalie-Score 7 Tage beobachten – bei Persistenz Ursachenanalyse",
    },

    # ── Antrieb / Motor ─────────────────────────────────────────────────────

    {
        "id": "KB-050",
        "name": "Erhöhte Motorlast – Antriebsstrang",
        "match_any": ["power_erhöht", "effizienz_sink"],
        "match_all": [],
        "exclude": ["temp_erhöht", "vib_erhöht"],
        "urgency": "mittel",
        "timeframe": "innerhalb 4 Stunden",
        "cause": (
            "Erhöhte Leistungsaufnahme ohne Temperatur- oder Vibrationsauffälligkeit "
            "deutet auf mechanische Verluste im Antriebsstrang hin: Getriebeverschleiß, "
            "Riemenspannung, Kupplungsdurchrutsch oder erhöhte Reibung in Führungen."
        ),
        "actions": [
            "Antriebsriemen auf Spannung und Verschleiß prüfen (lt. Handbuch-Sollwert)",
            "Getriebe auf Ölstand und Ölqualität prüfen",
            "Kupplung auf Spiel und Verschleiß inspizieren",
            "Linearführungen reinigen und nachschmieren",
            "Motorstrom im Betrieb messen und mit Nennstrom vergleichen",
        ],
        "monitoring": "Strom täglich loggen – Trend über 1 Woche auswerten",
    },

    {
        "id": "KB-051",
        "name": "Kritische Motorlast mit Vibrations-Leistungs-Kopplung",
        "match_any": [],
        "match_all": ["power_erhöht", "vib_pwr_kopplung"],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "innerhalb 1 Stunde",
        "cause": (
            "Veränderte Vibrations-Leistungs-Kopplung mit erhöhtem Energiebedarf deutet "
            "auf Unwucht oder Resonanzeffekte im Antriebsstrang hin. Diese verursachen "
            "Mehrbelastung des Motors und können zu Wicklungsschäden führen."
        ),
        "actions": [
            "Unwuchtmessung an Haupt-Rotationseinheit (Grenzwert: < G2.5 ISO 1940)",
            "Resonanzfrequenzen des Antriebsstrangs mit Schwingungsanalysator prüfen",
            "Fundamentschrauben auf Anzugsmoment prüfen",
            "Motorlüfter und -kühlung inspizieren",
        ],
        "monitoring": "Vibration + Strom jede 15 min; Produktionstakt auf 80 % reduzieren",
    },

    # ── Lager + Wärme kombiniert ────────────────────────────────────────────

    {
        "id": "KB-060",
        "name": "Überhitztes Lager – Lager-Wärme-Kombination",
        "match_any": [],
        "match_all": ["vib_erhöht", "temp_erhöht"],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "sofort prüfen",
        "cause": (
            "Gleichzeitige Vibrations- und Temperaturerhöhung ist klassisch für ein "
            "überhitztes Lager. Thermische Ausdehnung des Innenrings kann das Lager "
            "verspannen und die Vibration weiter verstärken – Eskalationsrisiko."
        ),
        "actions": [
            "Lagertemperatur direkt mit IR-Thermometer messen",
            "Bei Lagertemperatur > 80 °C: sofortige Notschmierung und Lastreduzierung",
            "Stoßimpulsmessung (SPM/HFD) an Lagerstelle",
            "Kühlluft auf Lagerbereich ausrichten",
            "Lager-Tauschplan für nächsten Stillstand vorbereiten",
        ],
        "monitoring": "Lagertemperatur + Vibration alle 10 min",
    },

    # ── Output-Probleme ohne klare Einzelursache ────────────────────────────

    {
        "id": "KB-070",
        "name": "Unerklärter Output-Rückgang",
        "match_any": ["output_gesunken"],
        "match_all": [],
        "exclude": ["vib_erhöht", "temp_erhöht", "hum_out_kopplung"],
        "urgency": "mittel",
        "timeframe": "innerhalb 2 Stunden",
        "cause": (
            "Output-Rückgang ohne klare mechanische oder thermische Begleitsymptome. "
            "Mögliche Ursachen: Materialzufuhrproblem, Werkzeugverschleiß, "
            "Bedienereingriff oder Prozessparameter-Drift."
        ),
        "actions": [
            "Materialzufuhr und Magazinfüllstand prüfen",
            "Werkzeugzustand begutachten (Verschleiß, Bruch, Positionierung)",
            "Bedienerprotokoll der letzten Schicht einsehen",
            "Prozessparameter gegen Sollwerte prüfen (Takt, Druck, Hubweg)",
            "Stichprobenprüfung der aktuellen Fertigteile (Maßhaltigkeit)",
        ],
        "monitoring": "Output-Rate alle 15 min bis zur Klärung",
    },

    # ── Wiederkehrende Anomalien ────────────────────────────────────────────

    {
        "id": "KB-080",
        "name": "Wiederkehrende Anomalien – systemische Grundursache",
        "match_any": ["vib_erhöht", "temp_erhöht", "output_gesunken", "effizienz_sink"],
        "match_all": [],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "vor nächster Schicht klären",
        "cause": (
            "Mehrfach aufgetretene Anomalien innerhalb kurzer Zeit deuten auf eine "
            "nicht behobene systemische Grundursache hin. Symptombehandlung allein "
            "reicht nicht – die Maschine hat ein strukturelles Problem."
        ),
        "actions": [
            "Root-Cause-Analyse (5-Why oder Ishikawa-Diagramm) einleiten",
            "Vorherige Wartungsberichte der letzten 30 Tage prüfen",
            "Maschinenparameter vollständig gegen Spezifikation prüfen",
            "Ungeplanten Wartungsstopp für die nächste Schicht einplanen",
            "Eskalation an Instandhaltungsleitung und Produktionsleitung",
        ],
        "monitoring": "Anomaliemuster über 3 Schichten dokumentieren",
    },

    # ── Kombinierte Komplexanomalien ────────────────────────────────────────

    {
        "id": "KB-090",
        "name": "Prozess- und Antriebsstörung kombiniert",
        "match_any": [],
        "match_all": ["hum_out_kopplung", "effizienz_sink"],
        "exclude": [],
        "urgency": "hoch",
        "timeframe": "sofort prüfen",
        "cause": (
            "Gleichzeitige Feuchtigkeitskopplung und gesunkene Effizienz deuten auf "
            "eine Wechselwirkung zwischen Materialverhalten und mechanischer Belastung hin. "
            "Feuchtes Material kann in Fördersystemen Kleben verursachen, was den Motor "
            "überlastet und die Effizienz senkt."
        ),
        "actions": [
            "Fördersystem auf Materialkleben oder -stau untersuchen",
            "Motorstrom auf Überlast prüfen",
            "Material-Trocknungssystem prüfen",
            "Kombinierten Wartungsstopp (Material + Antrieb) einplanen",
        ],
        "monitoring": "Strom + Output + Feuchte jede 10 min gemeinsam auswerten",
    },
]


# ── Lookup ──────────────────────────────────────────────────────────────────────

_URGENCY_ORDER = {"kritisch": 0, "hoch": 1, "mittel": 2, "niedrig": 3}


def lookup(tags: list[str]) -> list[dict]:
    """
    Sucht passende Einträge in der Wissensdatenbank für die gegebenen Symptom-Tags.
    Gibt Treffer sortiert nach Dringlichkeit zurück (kritisch zuerst).
    """
    matches = []
    for entry in KNOWLEDGE_BASE:
        if entry["match_all"] and not all(t in tags for t in entry["match_all"]):
            continue
        if entry["match_any"] and not any(t in tags for t in entry["match_any"]):
            continue
        if any(t in tags for t in entry.get("exclude", [])):
            continue
        matches.append(entry)

    matches.sort(key=lambda e: _URGENCY_ORDER.get(e["urgency"], 9))
    return matches
