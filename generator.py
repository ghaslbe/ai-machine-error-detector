#!/usr/bin/env python3
"""
Machine Simulator - generates a realistic multivariate time series
of an industrial production machine.

Physically motivated model:
  - temperatur   : thermal dynamics with setpoint, lag, production cycle
  - vibration    : mechanical, coupled to speed and bearing condition
  - humidity     : environmental, anticorrelated with temperature
  - output       : pieces/min - the main KPI
  - power_kw     : power consumption, correlated with output and temp load

Anomalies are injected into the physical state.
The detector has NO access to mode information.

Usage:
  python generator.py                      # demo scenario
  python generator.py --scenario normal    # always normal
  python generator.py --scenario manual    # read anomaly_mode.txt for live control
  python generator.py --interval 0.2      # faster sampling
"""

import csv
import time
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum


class AnomalyMode(IntEnum):
    NORMAL = 0
    BEARING_WEAR = 1    # slow vibration increase → power creep → output drop
    THERMAL_DRIFT = 2   # temperature setpoint drifts up → output slowly drops
    PROCESS_CHANGE = 3  # humidity-output coupling suddenly strengthens (e.g. material change)
    SENSOR_DRIFT = 4    # temperature sensor slowly drifts (calibration error)
    OVERHEAT = 5        # sudden heat event → temp exceeds rule threshold (both AI + rules fire)


@dataclass
class MachineState:
    """Internal physical state - not observable by the detector."""
    temp_setpoint: float = 65.0
    temp_actual: float = 65.0
    vibration_health: float = 0.0   # 0 = new, grows with bearing wear
    thermal_offset: float = 0.0    # setpoint drift
    sensor_offset: float = 0.0     # sensor calibration error
    step: int = 0


def simulate_step(state: MachineState, mode: AnomalyMode) -> dict:
    """
    One simulation step. Updates physical state and returns sensor readings.
    All correlations are physically motivated.
    """
    state.step += 1
    t = state.step

    # ── Anomaly injection into physical state ──────────────────────────────
    if mode == AnomalyMode.BEARING_WEAR:
        # +0.014/step → peaks at 1.1 in ~79 steps; vibration max ≈ 1.42 mm/s (rule: 1.60)
        # Deliberately close to the rule threshold to create dramatic but still AI-only episodes.
        state.vibration_health = min(state.vibration_health + 0.014, 1.1)
    elif mode == AnomalyMode.THERMAL_DRIFT:
        # +0.30/step → temp max ≈ 79°C with lag (rule: 82°C) – visibly close to threshold
        state.thermal_offset = min(state.thermal_offset + 0.30, 14.0)
    elif mode == AnomalyMode.SENSOR_DRIFT:
        # +0.09/step → 6°C offset after 67 steps; strong roc30 signal, clearly AI-detectable
        state.sensor_offset = min(state.sensor_offset + 0.09, 6.0)
    elif mode == AnomalyMode.OVERHEAT:
        state.thermal_offset = 25.0   # forces temp well above 82°C rule threshold
    elif mode == AnomalyMode.NORMAL:
        # Recovery toward baseline (represents maintenance / self-correction)
        state.vibration_health = max(0.0, state.vibration_health - 0.009)
        state.thermal_offset   = state.thermal_offset * 0.85   # faster: ~4 step half-life
        state.sensor_offset    = state.sensor_offset  * 0.85
    # PROCESS_CHANGE has no physical state change - only alters coupling coefficients

    # ── Temperature (first-order thermal model) ────────────────────────────
    # Production cycle: slight temp oscillation (~5 min period)
    temp_cycle = 2.2 * math.sin(2 * math.pi * t / 300)
    # Slow environmental drift (~60 min period)
    temp_env = 0.7 * math.sin(2 * math.pi * t / 3600)
    target_temp = state.temp_setpoint + state.thermal_offset + temp_cycle + temp_env
    # Thermal lag (time constant ~20 steps → sluggish response)
    state.temp_actual += (target_temp - state.temp_actual) / 20.0
    temp_noise = random.gauss(0, 0.12)
    # Sensor reads true value + calibration error + noise
    temp_measured = state.temp_actual + state.sensor_offset + temp_noise

    # ── Vibration (bearing + rotational speed) ─────────────────────────────
    # Normal: small amplitude oscillating with production speed
    vibration_base = 0.32 + 0.04 * math.sin(2 * math.pi * t / 180)
    vibration = vibration_base + state.vibration_health
    vibration += random.gauss(0, 0.018)
    vibration = max(0.05, vibration)

    # ── Humidity (environmental, anticorrelated with temp) ─────────────────
    humidity = 45.0 - 0.28 * (state.temp_actual - 65.0)
    humidity += 2.5 * math.sin(2 * math.pi * t / 1800)   # slow cycle
    humidity += random.gauss(0, 0.45)
    humidity = max(20.0, min(80.0, humidity))

    # ── Output (pieces/min) ───────────────────────────────────────────────
    # Physics: thermal expansion → slower at high temp
    #          vibration → mechanical slowdowns at high wear
    #          humidity → normally negligible, but strong in PROCESS_CHANGE
    output_base = 100.0
    temp_penalty = -0.22 * (state.temp_actual - 65.0)
    vib_penalty  = -4.0  * state.vibration_health
    if mode == AnomalyMode.PROCESS_CHANGE:
        # New coupling: humidity significantly affects output
        # (e.g. hygroscopic material change → swelling → slower feed)
        hum_penalty = -0.90 * (humidity - 45.0)
    else:
        hum_penalty = -0.07 * (humidity - 45.0)   # normally very weak

    output = output_base + temp_penalty + vib_penalty + hum_penalty
    output += random.gauss(0, 0.75)
    output = max(50.0, min(130.0, output))

    # ── Power consumption (kW) ─────────────────────────────────────────────
    # Driven by output rate, thermal load, and vibration losses
    power = 0.37 * output + 0.27 * state.temp_actual + 1.5 * state.vibration_health * output / 100
    power += random.gauss(0, 0.35)

    return {
        'timestamp':      t,
        'temperatur':     round(temp_measured, 3),
        'vibration':      round(vibration, 4),
        'humidity':       round(humidity, 3),
        'output':         round(output, 2),
        'power_kw':       round(power, 3),
        # Ground truth columns - written to CSV but detector ignores them
        '_mode':          int(mode),
        '_vib_health':    round(state.vibration_health, 4),
        '_temp_offset':   round(state.thermal_offset, 3),
        # Effect details for the generator's console output
        '_vib_penalty':   round(vib_penalty, 3),
        '_temp_penalty':  round(temp_penalty, 3),
        '_hum_penalty':   round(hum_penalty, 3),
    }


def _demo_schedule(step: int) -> AnomalyMode:
    """
    Timed injection schedule for the demo run.
    Detector never reads this function.

    Initial NORMAL phase covers 370 steps = 1.23 × production cycle (period 300).
    This ensures the Isolation Forest trains on one full cycle → stable model.

    Recovery gaps are kept at 110 steps minimum:
      - 100 steps cooldown + 10 step buffer for roc30 normalisation.
    At 0.3 s/step a 110-step gap = 33 s real time; the 3-min symptom window
    (600 steps = 3 min) therefore spans 2–3 previous anomaly episodes.
    """
    if   step <  370: return AnomalyMode.NORMAL           # training: one full production cycle
    elif step <  480: return AnomalyMode.BEARING_WEAR      # phase 1 – vib creeps to 1.42 mm/s
    elif step <  580: return AnomalyMode.NORMAL            # recovery (100 steps)
    elif step <  660: return AnomalyMode.THERMAL_DRIFT     # phase 2 – temp drifts to ~79 °C
    elif step <  760: return AnomalyMode.NORMAL            # recovery
    elif step <  830: return AnomalyMode.OVERHEAT          # phase 3 – temp > 82 °C, rules fire
    elif step <  930: return AnomalyMode.NORMAL            # recovery
    elif step < 1060: return AnomalyMode.PROCESS_CHANGE    # phase 4 – humidity-output coupling
    elif step < 1160: return AnomalyMode.NORMAL            # recovery
    elif step < 1300: return AnomalyMode.BEARING_WEAR      # phase 5 – longer, vib at max 1.42
    elif step < 1400: return AnomalyMode.NORMAL            # recovery
    elif step < 1470: return AnomalyMode.SENSOR_DRIFT      # phase 6 – 6 °C sensor offset
    elif step < 1570: return AnomalyMode.NORMAL            # recovery
    elif step < 1660: return AnomalyMode.THERMAL_DRIFT     # phase 7 – second thermal drift
    elif step < 1760: return AnomalyMode.NORMAL            # recovery
    elif step < 1890: return AnomalyMode.PROCESS_CHANGE    # phase 8 – second process change
    elif step < 1990: return AnomalyMode.NORMAL            # recovery
    elif step < 2120: return AnomalyMode.BEARING_WEAR      # phase 9 – longest wear episode
    elif step < 2220: return AnomalyMode.NORMAL            # recovery
    elif step < 2290: return AnomalyMode.OVERHEAT          # phase 10 – second overheat
    else:             return AnomalyMode.NORMAL


def _read_manual_mode() -> AnomalyMode:
    try:
        val = int(Path('anomaly_mode.txt').read_text().strip())
        return AnomalyMode(val)
    except Exception:
        return AnomalyMode.NORMAL


def _format_anomaly_detail(row: dict, state: MachineState, mode: AnomalyMode) -> str:
    """
    Produces a human-readable explanation of what the current anomaly is doing
    to the machine - shown at the end of the generator's console line.
    """
    if mode == AnomalyMode.NORMAL:
        return ''

    parts = []
    if mode == AnomalyMode.BEARING_WEAR:
        wear  = state.vibration_health
        extra_vib = wear                         # mm/s added to baseline
        out_loss  = abs(row['_vib_penalty'])
        parts.append(f"Lagerverschleiß: +{extra_vib:.3f}mm/s Vibration")
        parts.append(f"Output -{out_loss:.1f}/min")

    elif mode == AnomalyMode.THERMAL_DRIFT:
        offset   = state.thermal_offset
        out_loss = abs(row['_temp_penalty'])
        parts.append(f"Thermaldrift: Sollwert +{offset:.2f}°C")
        parts.append(f"Output -{out_loss:.1f}/min durch Wärmeausdehnung")

    elif mode == AnomalyMode.PROCESS_CHANGE:
        hum_effect = row['_hum_penalty']
        parts.append(f"Prozessänderung: Feuchtekopplung aktiv")
        parts.append(f"Feuchtigkeit wirkt {hum_effect:+.1f}/min auf Output")

    elif mode == AnomalyMode.SENSOR_DRIFT:
        offset = state.sensor_offset
        parts.append(f"Sensordrift: Temp-Fühler +{offset:.2f}°C Offset")
        parts.append(f"Istwert liegt bei {row['temperatur'] - offset:.2f}°C")

    elif mode == AnomalyMode.OVERHEAT:
        parts.append(f"ÜBERHITZUNG: Temp {row['temperatur']:.1f}°C >> Grenzwert 82°C")
        out_loss = abs(row['_temp_penalty'])
        parts.append(f"Output -{out_loss:.1f}/min")

    return '  ⚠ ' + ' | '.join(parts)


def run_generator(output_file: str, interval: float, scenario: str):
    path = Path(output_file)
    fieldnames = [
        'timestamp', 'temperatur', 'vibration', 'humidity', 'output', 'power_kw',
        '_mode', '_vib_health', '_temp_offset', '_vib_penalty', '_temp_penalty', '_hum_penalty',
    ]

    write_header = not path.exists() or path.stat().st_size == 0
    state = MachineState()

    print(f"[Generator] Output → {output_file}  |  interval={interval}s  |  scenario={scenario}")
    print(f"[Generator] Anomaly modes: 0=Normal 1=BearingWear 2=ThermalDrift "
          f"3=ProcessChange 4=SensorDrift")
    print()

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        step = 0
        while True:
            if scenario == 'normal':
                mode = AnomalyMode.NORMAL
            elif scenario == 'manual':
                mode = _read_manual_mode()
            else:
                mode = _demo_schedule(step)

            row = simulate_step(state, mode)
            writer.writerow(row)
            f.flush()

            detail = _format_anomaly_detail(row, state, mode)
            print(
                f"[{row['timestamp']:5d}] "
                f"T={row['temperatur']:6.2f}°C  "
                f"V={row['vibration']:.4f}mm/s  "
                f"H={row['humidity']:5.1f}%  "
                f"Out={row['output']:6.2f}/min  "
                f"P={row['power_kw']:5.1f}kW"
                f"{detail}"
            )

            step += 1
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Industrial Machine Simulator')
    parser.add_argument('--out',      default='stream.csv')
    parser.add_argument('--interval', type=float, default=0.3,
                        help='Seconds between samples (default 0.3)')
    parser.add_argument('--scenario', choices=['demo', 'normal', 'manual'], default='demo')
    args = parser.parse_args()

    try:
        run_generator(args.out, args.interval, args.scenario)
    except KeyboardInterrupt:
        print('\n[Generator] Stopped.')
