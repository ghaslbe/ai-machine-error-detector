#!/usr/bin/env python3
"""
Orchestrator + Live Visualisation

Runs generator and detector in separate threads and displays a live
4-panel matplotlib plot:
  1. Sensor signals (temp, vibration, output, humidity, power)
  2. AI anomaly score + threshold
  3. Rule-based alerts (per-sensor bar)
  4. Comparison: AI-only vs Rule-only detections

Usage:
  python main.py                  # full demo with live plot
  python main.py --no-plot        # headless (terminal output only)
  python main.py --interval 0.2   # faster
"""

import threading
import time
import argparse
import subprocess
import sys
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

STREAM_FILE = 'stream.csv'
PLOT_WINDOW = 200   # how many samples to show in the live plot


def run_generator(interval: float, scenario: str):
    """Run generator in subprocess so it is fully independent."""
    cmd = [sys.executable, 'generator.py',
           '--out', STREAM_FILE,
           '--interval', str(interval),
           '--scenario', scenario]
    proc = subprocess.Popen(cmd)
    return proc


def run_detector_thread(result_queue, stop_event):
    """Run the detector logic in a thread, push results into result_queue."""
    import csv
    import time as _time
    from detector import (
        StreamReader, IsolationForestDetector, check_rules,
        SENSOR_COLS, AI_THRESHOLD, TRAIN_SAMPLES
    )

    path = Path(STREAM_FILE)
    while not path.exists():
        _time.sleep(0.1)
    _time.sleep(0.3)

    reader = StreamReader(STREAM_FILE)
    ai_det = IsolationForestDetector()
    trained = False
    count = 0

    for raw_row in reader.follow():
        if stop_event.is_set():
            break

        try:
            sensor_vals = {k: float(raw_row[k]) for k in SENSOR_COLS}
            sensor_vals['timestamp'] = int(raw_row['timestamp'])
            # ground truth for plot (not used by detector logic)
            sensor_vals['_mode'] = int(raw_row.get('_mode', 0))
        except (ValueError, KeyError):
            continue

        rule_violations = check_rules(sensor_vals)
        rule_alert = len(rule_violations) > 0

        ai_result = ai_det.ingest(raw_row)

        if ai_result is None:
            n = len(ai_det.train_buffer)
            result_queue.append({
                'timestamp': sensor_vals.get('timestamp', count),
                'sensors': sensor_vals,
                'ai_score': None,
                'ai_alert': False,
                'rule_alert': rule_alert,
                'rule_violations': rule_violations,
                'training': True,
                '_mode': sensor_vals['_mode'],
            })
            count += 1
            continue

        if not trained and ai_det.trained:
            trained = True

        result_queue.append({
            'timestamp':       ai_result['timestamp'],
            'sensors':         sensor_vals,
            'ai_score':        ai_result['score'],
            'ai_alert':        ai_result['is_alert'],
            'rule_alert':      rule_alert,
            'rule_violations': rule_violations,
            'training':        False,
            '_mode':           sensor_vals['_mode'],
        })
        count += 1


def live_plot(result_queue: deque, stop_event: threading.Event):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle('Maschinenüberwachung: KI vs. Regelbasiert', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    ax_sig, ax_ai, ax_rule, ax_cmp = axes

    # History buffers
    N = PLOT_WINDOW
    ts      = deque(maxlen=N)
    temps   = deque(maxlen=N)
    vibs    = deque(maxlen=N)
    outputs = deque(maxlen=N)
    powers  = deque(maxlen=N)
    ai_scores   = deque(maxlen=N)
    ai_alerts   = deque(maxlen=N)
    rule_alerts = deque(maxlen=N)
    modes       = deque(maxlen=N)
    training_flags = deque(maxlen=N)

    MODE_NAMES = {0: 'Normal', 1: 'Lagerverschleiß',
                  2: 'Thermaldrift', 3: 'Prozessänderung', 4: 'Sensordrift'}
    MODE_COLORS = {0: '#2ecc71', 1: '#e67e22', 2: '#e74c3c',
                   3: '#9b59b6', 4: '#1abc9c'}

    def update(_frame):
        while result_queue:
            item = result_queue.popleft() if hasattr(result_queue, 'popleft') else result_queue.pop(0)
            ts.append(item['timestamp'])
            s = item['sensors']
            temps.append(s.get('temperatur', np.nan))
            vibs.append(s.get('vibration', np.nan) * 100)  # scale for visibility
            outputs.append(s.get('output', np.nan))
            powers.append(s.get('power_kw', np.nan))
            ai_scores.append(item['ai_score'])
            ai_alerts.append(item['ai_alert'])
            rule_alerts.append(item['rule_alert'])
            modes.append(item['_mode'])
            training_flags.append(item['training'])

        if len(ts) < 2:
            return

        t_arr = np.array(ts)
        m_arr = np.array(modes)
        tr_arr = np.array(training_flags)

        # ── Panel 1: Sensor signals ────────────────────────────────────────
        ax_sig.clear()
        ax_sig.plot(t_arr, list(temps),   label='Temperatur (°C)', color='#e74c3c', lw=1.3)
        ax_sig.plot(t_arr, list(outputs), label='Output (Stück/min)', color='#2980b9', lw=1.3)
        ax_sig.plot(t_arr, list(powers),  label='Power (kW)', color='#8e44ad', lw=1.0, alpha=0.7)
        ax_sig.plot(t_arr, list(vibs),    label='Vibration ×100', color='#e67e22', lw=1.0, alpha=0.8)

        # Background colour for anomaly phases
        _shade_modes(ax_sig, t_arr, m_arr, tr_arr)
        ax_sig.set_ylabel('Sensorwerte')
        ax_sig.legend(loc='upper left', fontsize=7, ncol=2)
        ax_sig.grid(True, alpha=0.3)

        # ── Panel 2: AI score ──────────────────────────────────────────────
        ax_ai.clear()
        sc_arr = np.array([v if v is not None else np.nan for v in ai_scores])
        ax_ai.plot(t_arr, sc_arr, color='#2980b9', lw=1.4, label='AI Score (Isolation Forest)')
        ax_ai.axhline(y=-0.15, color='red', linestyle='--', lw=1, label=f'Schwelle ({AI_THRESHOLD})')

        # Fill anomaly band
        ax_ai.fill_between(t_arr, sc_arr, -0.15,
                           where=sc_arr < -0.15,
                           color='red', alpha=0.25, label='KI-Alarm')
        _shade_modes(ax_ai, t_arr, m_arr, tr_arr)
        ax_ai.set_ylabel('Anomalie-Score')
        ax_ai.set_ylim(-0.5, 0.1)
        ax_ai.legend(loc='upper left', fontsize=7)
        ax_ai.grid(True, alpha=0.3)

        # Training zone marker
        if tr_arr.any():
            last_train = t_arr[tr_arr].max() if tr_arr.any() else t_arr[0]
            ax_ai.axvline(x=last_train, color='gray', linestyle=':', lw=1)
            ax_ai.text(last_train, -0.48, 'Training Ende', fontsize=6, color='gray')

        # ── Panel 3: Rule-based alerts ─────────────────────────────────────
        ax_rule.clear()
        ra_arr = np.array(rule_alerts, dtype=float)
        ax_rule.fill_between(t_arr, 0, ra_arr, step='mid',
                             color='#e74c3c', alpha=0.7, label='Regelbasierter Alarm')
        _shade_modes(ax_rule, t_arr, m_arr, tr_arr)
        ax_rule.set_ylabel('Regelverstoß')
        ax_rule.set_ylim(-0.1, 1.5)
        ax_rule.set_yticks([0, 1])
        ax_rule.set_yticklabels(['OK', 'ALARM'])
        ax_rule.legend(loc='upper left', fontsize=7)
        ax_rule.grid(True, alpha=0.3)

        # ── Panel 4: Comparison ────────────────────────────────────────────
        ax_cmp.clear()
        ai_arr   = np.array(ai_alerts,   dtype=float)
        rule_arr = np.array(rule_alerts, dtype=float)

        # AI-only (yellow): AI detects, rules miss
        ai_only_mask  = ai_arr * (1 - rule_arr)
        rule_only_mask = rule_arr * (1 - ai_arr)
        both_mask      = ai_arr * rule_arr

        ax_cmp.fill_between(t_arr, 0, ai_only_mask * 1.0, step='mid',
                            color='#f39c12', alpha=0.85, label='Nur KI erkennt')
        ax_cmp.fill_between(t_arr, 0, rule_only_mask * 0.6, step='mid',
                            color='#e74c3c', alpha=0.7, label='Nur Regelwerk')
        ax_cmp.fill_between(t_arr, 0, both_mask * 0.8, step='mid',
                            color='#8e44ad', alpha=0.6, label='Beide')

        _shade_modes(ax_cmp, t_arr, m_arr, tr_arr)
        ax_cmp.set_ylabel('Vergleich')
        ax_cmp.set_ylim(-0.1, 1.3)
        ax_cmp.set_yticks([])
        ax_cmp.set_xlabel('Zeitschritt')
        ax_cmp.legend(loc='upper left', fontsize=7, ncol=3)
        ax_cmp.grid(True, alpha=0.3)

        # Legend for mode colours
        if len(set(m_arr)) > 1:
            patches = [mpatches.Patch(color=MODE_COLORS.get(m, 'gray'),
                                      alpha=0.2, label=MODE_NAMES.get(m, str(m)))
                       for m in sorted(set(m_arr))]
            ax_sig.legend(handles=list(ax_sig.get_lines()) + patches,
                          loc='upper left', fontsize=6, ncol=3)

        fig.canvas.draw_idle()

    def _shade_modes(ax, t_arr, m_arr, tr_arr):
        """Draw faint background shading for anomaly phases."""
        if len(t_arr) < 2:
            return
        prev_m = m_arr[0]
        start  = t_arr[0]
        for i in range(1, len(t_arr)):
            if m_arr[i] != prev_m or i == len(t_arr) - 1:
                c = MODE_COLORS.get(int(prev_m), '#aaa')
                ax.axvspan(start, t_arr[i], alpha=0.08,
                           color=c, linewidth=0)
                prev_m = m_arr[i]
                start  = t_arr[i]

    ani = FuncAnimation(fig, update, interval=400, cache_frame_data=False)
    plt.show()
    stop_event.set()


def main():
    parser = argparse.ArgumentParser(description='KI-Maschinenüberwachung')
    parser.add_argument('--interval', type=float, default=0.3)
    parser.add_argument('--scenario', choices=['demo', 'normal', 'manual'], default='demo')
    parser.add_argument('--no-plot', action='store_true', help='Headless mode')
    args = parser.parse_args()

    # Clear old stream
    stream = Path(STREAM_FILE)
    if stream.exists():
        stream.unlink()

    stop_event = threading.Event()

    # result_queue shared between detector thread and plot
    from collections import deque as _deque
    result_queue = _deque()

    # Start generator subprocess
    gen_proc = run_generator(args.interval, args.scenario)

    # Start detector thread
    det_thread = threading.Thread(
        target=run_detector_thread,
        args=(result_queue, stop_event),
        daemon=True,
    )
    det_thread.start()

    print("[Main] Generator and detector running.")
    print("[Main] Press Ctrl+C to stop.\n")

    if args.no_plot:
        # Headless: just print from queue
        try:
            while True:
                while result_queue:
                    item = result_queue.popleft()
                    if item['training']:
                        continue
                    s = item['sensors']
                    ai_tag   = 'ALARM' if item['ai_alert']   else '  ok '
                    rule_tag = 'ALARM' if item['rule_alert'] else '  ok '
                    print(
                        f"[{item['timestamp']:5d}] "
                        f"T={s.get('temperatur',0):6.2f} "
                        f"V={s.get('vibration',0):.4f} "
                        f"H={s.get('humidity',0):5.1f} "
                        f"Out={s.get('output',0):6.2f} | "
                        f"AI:{ai_tag}  Rules:{rule_tag}"
                    )
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    else:
        try:
            live_plot(result_queue, stop_event)
        except KeyboardInterrupt:
            pass

    stop_event.set()
    gen_proc.terminate()
    print('\n[Main] Done.')


if __name__ == '__main__':
    main()
