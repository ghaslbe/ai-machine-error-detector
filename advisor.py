#!/usr/bin/env python3
"""
SymptomTracker + LLM-Advisor

SymptomTracker
  Verfolgt alle Alarm-Ereignisse der letzten SYMPTOM_HISTORY_WINDOW Schritte
  (entspricht ~3 Minuten bei 0.3 s/Schritt = 600 Schritte).
  Ermöglicht es, mehrere aufeinanderfolgende Anomalie-Episoden gemeinsam
  darzustellen und dem LLM als Kontext zu übergeben.

LLMAdvisor
  Baut aus der Symptomhistorie, dem aktuellen Ereignis und passenden
  Wissensdatenbank-Einträgen einen Prompt und ruft ein LLM via OpenRouter auf.
  Der API-Aufruf läuft in einem Hintergrund-Thread, damit der Detektor-Loop
  nicht blockiert wird.
"""

import os
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import requests

from knowledge_db import match_symptoms, lookup, SYMPTOM_LABELS


# ── Konfiguration ───────────────────────────────────────────────────────────────

def _load_env():
    """Lädt .env aus dem Projektverzeichnis, falls vorhanden."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_env()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL      = "qwen/qwen3-30b-a3b"   # OpenRouter model ID


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY nicht gesetzt. "
            "Bitte in .env eintragen: OPENROUTER_API_KEY=sk-or-v1-..."
        )
    return key
LLM_MAX_TOKENS   = 500
LLM_TEMPERATURE  = 0.3

# 3 Minuten Maschinenzeit @ 0.3 s/Schritt = 600 Schritte
SYMPTOM_HISTORY_WINDOW = 600

URGENCY_EMOJI = {"kritisch": "🚨", "hoch": "⚠️", "mittel": "🔶", "niedrig": "🔵"}
LINE = "─" * 64


# ── Datenstrukturen ─────────────────────────────────────────────────────────────

@dataclass
class SymptomEvent:
    timestamp:       int
    explanation:     str          # aus _explain()
    tags:            list[str]    # geparste Symptom-Tags
    rule_violations: list[str]
    ai_alert:        bool
    rule_alert:      bool
    sensor_values:   dict         # Momentaufnahme der Sensorwerte


# ── SymptomTracker ──────────────────────────────────────────────────────────────

class SymptomTracker:
    """
    Rollendes Fenster der letzten SYMPTOM_HISTORY_WINDOW Schritte.
    Speichert nur Schritte, in denen mindestens ein Alert aktiv war.
    """

    def __init__(self, window: int = SYMPTOM_HISTORY_WINDOW):
        self._window = window
        self._events: deque[SymptomEvent] = deque()

    # ── Schreiben ───────────────────────────────────────────────────────────

    def add(self, timestamp: int, explanation: str | None,
            rule_violations: list[str], ai_alert: bool, rule_alert: bool,
            sensor_values: dict) -> SymptomEvent:
        """
        Neues Symptom-Ereignis aufzeichnen.
        Gibt das erstellte SymptomEvent zurück (wird vom Aufrufer für den LLM-Call benötigt).
        """
        expl = explanation or ""
        tags = match_symptoms(expl)
        ev = SymptomEvent(
            timestamp       = timestamp,
            explanation     = expl,
            tags            = tags,
            rule_violations = list(rule_violations),
            ai_alert        = ai_alert,
            rule_alert      = rule_alert,
            sensor_values   = dict(sensor_values),
        )
        self._events.append(ev)
        # Alte Einträge außerhalb des Fensters entfernen
        cutoff = timestamp - self._window
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()
        return ev

    # ── Lesen ───────────────────────────────────────────────────────────────

    def get_recent(self, current_ts: int) -> list[SymptomEvent]:
        """Alle Ereignisse der letzten WINDOW Schritte."""
        cutoff = current_ts - self._window
        return [e for e in self._events if e.timestamp >= cutoff]

    def get_all_tags(self, current_ts: int) -> list[str]:
        """Alle einzigartigen Symptom-Tags aus der jüngsten Geschichte."""
        seen: set[str] = set()
        for ev in self.get_recent(current_ts):
            seen.update(ev.tags)
        return list(seen)

    # ── Formatierung ────────────────────────────────────────────────────────

    def format_history(self, current_ts: int) -> str:
        """
        Formatiert die Symptomhistorie als lesbaren Block.
        Gruppiert aufeinanderfolgende Schritte derselben Episode (Lücke ≤ 60 Schritte).
        Zeigt je Episode eine Zeile mit Zeitbereich + dominanter Erklärung.
        """
        events = self.get_recent(current_ts)
        if not events:
            return "  (keine Symptomhistorie vorhanden)"

        # Gruppierung in Episoden
        episodes: list[list[SymptomEvent]] = []
        current_ep = [events[0]]
        for ev in events[1:]:
            if ev.timestamp - current_ep[-1].timestamp > 60:
                episodes.append(current_ep)
                current_ep = [ev]
            else:
                current_ep.append(ev)
        episodes.append(current_ep)

        lines = []
        for ep_idx, ep in enumerate(episodes):
            t0 = ep[0].timestamp
            t1 = ep[-1].timestamp

            # Marker
            has_both  = any(e.ai_alert and e.rule_alert for e in ep)
            has_ai    = any(e.ai_alert for e in ep)
            has_rule  = any(e.rule_alert for e in ep)
            if has_both:
                marker = "🔴 KI+Regel"
            elif has_ai:
                marker = "🟡 Nur KI  "
            elif has_rule:
                marker = "🔴 Nur Reg."
            else:
                marker = "⚪ Info    "

            # Aussagekräftigste Erklärung der Episode (längste)
            explanations = [e.explanation for e in ep if e.explanation]
            best_expl = max(explanations, key=len) if explanations else "—"
            best_expl = best_expl[:55] + "…" if len(best_expl) > 55 else best_expl

            # Tags sammeln
            ep_tags = list({t for e in ep for t in e.tags})
            tag_labels = [SYMPTOM_LABELS.get(t, t) for t in ep_tags[:3]]
            tag_str = ", ".join(tag_labels) if tag_labels else "—"

            if t0 == t1:
                header = f"  [{t0:5d}]       {marker}  {best_expl}"
            else:
                header = f"  [{t0:5d}–{t1:5d}] {marker}  {best_expl}"
            lines.append(header)

            # Regelverstoß in Episode
            rule_msgs = list({v for e in ep for v in e.rule_violations})
            if rule_msgs:
                lines.append(f"    {'':5}  Regelverstoß: {', '.join(rule_msgs[:2])}")

        return "\n".join(lines)

    def print_history_block(self, current_ts: int):
        """Gibt den Symptom-Verlauf als formatierten Kasten aus."""
        events = self.get_recent(current_ts)
        if len(events) <= 1:
            return   # nur das aktuelle Ereignis – keine Historie zu zeigen

        span_start = events[0].timestamp
        real_min   = (current_ts - span_start) * 0.3 / 60  # echte Minuten bei 0.3 s/Schritt

        print(f"\n\033[35m┌{LINE}┐\033[0m")
        print(f"\033[35m│  📋 SYMPTOM-VERLAUF  "
              f"(Schritte {span_start}–{current_ts}, ≈ {real_min:.1f} min)"
              f"{' ' * max(0, 64 - 42 - len(str(span_start)) - len(str(current_ts)))}│\033[0m")
        print(f"\033[35m├{LINE}┤\033[0m")
        for line in self.format_history(current_ts).splitlines():
            padded = f"{line:<64}"[:64]
            print(f"\033[35m│{padded}│\033[0m")
        print(f"\033[35m└{LINE}┘\033[0m")


# ── LLMAdvisor ─────────────────────────────────────────────────────────────────

class LLMAdvisor:
    """
    Baut einen kontextreichen Prompt aus Symptomhistorie + Wissensdatenbank
    und ruft asynchron ein LLM via OpenRouter auf.

    Ausgabe erfolgt in einem farbigen Kasten im Terminal, nachdem der API-Call
    abgeschlossen ist – der Detektor-Loop läuft währenddessen weiter.
    """

    def __init__(self, model: str = LLM_MODEL):
        self._key   = _get_api_key()
        self._model = model
        self._busy  = False   # verhindert parallele LLM-Calls

    def query_async(self, tracker: SymptomTracker, current_ts: int,
                    event: SymptomEvent):
        """LLM-Anfrage in Hintergrund-Thread starten."""
        if self._busy:
            return
        self._busy = True
        t = threading.Thread(
            target=self._run,
            args=(tracker, current_ts, event),
            daemon=True,
        )
        t.start()

    # ── Interna ─────────────────────────────────────────────────────────────

    def _run(self, tracker: SymptomTracker, current_ts: int, event: SymptomEvent):
        try:
            response = self._call_api(tracker, current_ts, event)
            self._print_response(response, current_ts)
        except requests.exceptions.Timeout:
            print(f"\n  [LLM-Advisor] Timeout – API hat nicht rechtzeitig geantwortet.\n")
        except requests.exceptions.ConnectionError:
            print(f"\n  [LLM-Advisor] Verbindungsfehler – kein Internetzugang?\n")
        except Exception as exc:
            print(f"\n  [LLM-Advisor] Fehler: {exc}\n")
        finally:
            self._busy = False

    def _build_prompt(self, tracker: SymptomTracker, current_ts: int,
                      event: SymptomEvent) -> str:
        sv = event.sensor_values
        sensor_block = (
            f"  Temperatur:        {sv.get('temperatur', '?'):.2f} °C\n"
            f"  Vibration:         {sv.get('vibration', '?'):.4f} mm/s\n"
            f"  Luftfeuchtigkeit:  {sv.get('humidity', '?'):.1f} %\n"
            f"  Output:            {sv.get('output', '?'):.2f} Stück/min\n"
            f"  Leistungsaufnahme: {sv.get('power_kw', '?'):.1f} kW"
        )

        history_text = tracker.format_history(current_ts)

        # Tags aus gesamter Historie + aktuellem Ereignis
        combined_tags = list(set(event.tags + tracker.get_all_tags(current_ts)))
        kb_entries    = lookup(combined_tags)

        if kb_entries:
            kb_parts = []
            for e in kb_entries[:4]:   # max 4 Einträge im Kontext
                emoji = URGENCY_EMOJI.get(e["urgency"], "•")
                action_lines = "\n".join(f"    - {a}" for a in e["actions"])
                kb_parts.append(
                    f"[{e['id']}] {emoji} {e['name']} "
                    f"({e['urgency'].upper()}, {e['timeframe']})\n"
                    f"  Ursache: {e['cause']}\n"
                    f"  Maßnahmen:\n{action_lines}"
                )
            kb_block = "\n\n".join(kb_parts)
        else:
            kb_block = "Keine spezifischen Einträge gefunden – allgemeine Analyse erforderlich."

        rule_line = (
            f"Regelverstoß: {', '.join(event.rule_violations)}"
            if event.rule_violations
            else "Regelbasierte Grenzwerte: alle eingehalten"
        )

        prompt = f"""Du bist ein erfahrener Industriemaschinen-Instandhaltungsexperte.
Eine KI-Überwachung (Isolation Forest) meldet eine Anomalie an einer Produktionsmaschine.

=== AKTUELLE SENSORWERTE (Zeitschritt {current_ts}) ===
{sensor_block}

=== KI-DIAGNOSE (aktuell) ===
{event.explanation or 'Kein Erklärungstext – multivariates Muster.'}
{rule_line}

=== SYMPTOM-VERLAUF (letzte 3 Minuten Maschinenzeit) ===
{history_text}

=== RELEVANTE WISSENSDATENBANK-EINTRÄGE ===
{kb_block}

=== DEINE AUFGABE ===
Analysiere die Situation – vor allem den VERLAUF der letzten 3 Minuten – und gib eine
kompakte, praxisnahe Empfehlung in folgender Struktur:

1. URSACHE: Was ist die wahrscheinlichste Ursache dieser Anomalie?
2. SOFORTMASSNAHMEN: Nenne 3 konkrete Sofortmaßnahmen (nummeriert).
3. DRINGLICHKEIT: sofort / innerhalb 1h / innerhalb 8h – und warum?
4. FOLGERISIKO: Was passiert, wenn nichts unternommen wird?

Antworte auf Deutsch. Präzise, praxisorientiert, maximal 250 Wörter."""
        return prompt

    def _call_api(self, tracker: SymptomTracker, current_ts: int,
                  event: SymptomEvent) -> str:
        prompt  = self._build_prompt(tracker, current_ts, event)
        payload = {
            "model":    self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens":  LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
        }
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://kisimulation.local",
                "X-Title":       "KI-Maschinenüberwachung Demo",
            },
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _print_response(self, response: str, ts: int):
        model_short = self._model.split("/")[-1][:28]
        print(f"\n\033[36m┌{LINE}┐\033[0m")
        print(f"\033[36m│  🤖 KI-BERATER  ({model_short})  [Schritt {ts}]"
              f"{' ' * max(0, 64 - 18 - len(model_short) - len(str(ts)))}│\033[0m")
        print(f"\033[36m├{LINE}┤\033[0m")

        for para in response.strip().split("\n"):
            para = para.rstrip()
            if not para:
                print(f"\033[36m│{' ' * 64}│\033[0m")
                continue
            # Zeilenumbruch bei > 62 Zeichen
            while len(para) > 62:
                cut = para.rfind(" ", 0, 62)
                if cut == -1:
                    cut = 62
                print(f"\033[36m│  {para[:cut]:<62}│\033[0m")
                para = para[cut:].lstrip()
            if para:
                print(f"\033[36m│  {para:<62}│\033[0m", flush=True)

        print(f"\033[36m└{LINE}┘\033[0m\n", flush=True)
