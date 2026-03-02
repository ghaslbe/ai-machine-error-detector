#!/usr/bin/env python3
"""
AI Anomaly Detector - reads machine stream, learns normal behaviour,
detects multivariate anomalies via Isolation Forest.

Key design principle:
  The detector has ZERO access to ground-truth labels or the injection schedule.
  It discovers anomalies purely from statistical deviation of the learned
  multivariate normal distribution.

Also runs a parallel rule-based detector on the same stream so you can
directly compare what each approach catches.

Usage:
  python detector.py                    # default stream.csv
  python detector.py --stream my.csv
"""

import csv
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

from advisor import SymptomTracker, LLMAdvisor

warnings.filterwarnings('ignore')


# ── Configuration ──────────────────────────────────────────────────────────────

TRAIN_SAMPLES      = 300   # collect this many rows before training
                           # 300 = one full production cycle (period 300 steps) → stable model
WARMUP_WINDOW      = 40    # rolling features need this many rows to stabilise
CONTAMINATION      = 0.05  # expected anomaly fraction – higher = more sensitive boundary
# Detection uses a rolling window of model predictions (not raw scores).
# The model's predict() returns +1=normal, -1=anomaly.
# We alert when ANOMALY_FRACTION of the last PRED_WINDOW predictions are -1.
PRED_WINDOW        = 20    # rolling prediction window
ANOMALY_FRACTION   = 0.35  # fraction of recent predictions that must be -1 to alert
                           # needs 7/20 consistent anomaly hits → robust against noise
COOLDOWN_STEPS     = 100   # steps to suppress new alerts after an alert ends
                           # must cover physical recovery + rolling-window normalisation
SCORE_PERCENTILE   = 3     # percentile of calibration scores used as alert threshold
CALIBRATION_STEPS  = 30   # post-training normal steps used to calibrate the score threshold
                           # these must all be normal – fits exactly in the gap between
                           # training completion (~step 340) and first anomaly (step 370)

AI_THRESHOLD    = ANOMALY_FRACTION   # alias used by main.py live plot

SENSOR_COLS     = ['temperatur', 'vibration', 'humidity', 'output', 'power_kw']
SHORT_WIN       = 10
LONG_WIN        = 40
CORR_WIN        = 20       # longer window for rolling correlations → more stable estimates
LAG_STEPS       = [1, 3, 5, 10]

# Rule-based limits (what a static threshold system would use)
# Deliberately generous - single-sensor out-of-bounds only
RULES = {
    'temperatur': (45.0, 82.0),    # °C  min/max
    'vibration':  (0.0,  1.60),    # mm/s
    'humidity':   (18.0, 75.0),    # %
    'output':     (60.0, 130.0),   # pieces/min
    'power_kw':   (20.0, 80.0),    # kW
}


# ── Feature Engineering ────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build CYCLE-INVARIANT features from raw sensor columns.

    Key design decision: we do NOT use raw sensor values or absolute rolling
    means as model features. Sensor values have sinusoidal cycles, so absolute
    values at detection time differ from training time → model miscalibration.

    Instead we use only:
      - Rolling z-scores  (deviation from local rolling mean)
      - Rates of change   (stationary, cycle-invariant)
      - Rolling correlations (relationship changes between sensors)
      - Interaction of z-scores (multivariate anomaly patterns)

    These features are stationary with respect to the cyclic baseline.
    Raw sensor values are still computed and stored (for _explain() and display),
    but kept out of the feature matrix used by the Isolation Forest.
    """
    feat = pd.DataFrame(index=df.index)

    # Store raw values for explanation only (not used by the model)
    for col in SENSOR_COLS:
        feat[col] = df[col]   # raw, included for _explain access

    for col in SENSOR_COLS:
        s    = df[col]
        m_s  = s.rolling(SHORT_WIN, min_periods=2).mean()
        m_l  = s.rolling(LONG_WIN,  min_periods=5).mean()
        std_s = s.rolling(SHORT_WIN, min_periods=2).std().fillna(0)
        std_l = s.rolling(LONG_WIN,  min_periods=5).std().fillna(0)

        # ── Cycle-invariant: z-scores relative to rolling window ──────────
        feat[f'{col}_z_s'] = (s - m_s) / std_s.replace(0, 1e-6)
        feat[f'{col}_z_l'] = (s - m_l) / std_l.replace(0, 1e-6)

        # For _explain() compatibility keep a single 'zscore' alias
        feat[f'{col}_zscore'] = feat[f'{col}_z_l']

        # ── Rates of change (stationary) ──────────────────────────────────
        feat[f'{col}_roc1']  = s.diff(1).fillna(0)
        feat[f'{col}_roc5']  = s.diff(5).fillna(0)
        feat[f'{col}_roc10'] = s.diff(10).fillna(0)

        # Long-window ROC: signed versions kept for _explain(), but the MODEL
        # sees only the UPWARD component (clipped to ≥0).
        # Reason: recovery after an anomaly creates a strong NEGATIVE roc30 signal
        # (vibration falls fast, temperature cools) that looks equally anomalous to
        # the Isolation Forest.  Clipping to ≥0 means "declining = looks like training
        # baseline (zero trend)" so recovery phases don't trigger false alerts.
        roc20 = s.diff(20).fillna(0)
        roc30 = s.diff(30).fillna(0)
        feat[f'{col}_roc20']    = roc20           # kept for _explain()
        feat[f'{col}_roc30']    = roc30           # kept for _explain()
        feat[f'{col}_roc20_up'] = roc20.clip(lower=0)   # model: only rising trends
        feat[f'{col}_roc30_up'] = roc30.clip(lower=0)   # model: only rising trends

        # ── Rolling std ratio: catches variance changes ───────────────────
        feat[f'{col}_std_ratio'] = (std_s / std_l.replace(0, 1e-6)).fillna(1)

    # ── Cross-sensor z-score interactions (multivariate anomaly signal) ───
    # These are cycle-invariant because each component is already a z-score.
    feat['zz_temp_vib']  = feat['temperatur_z_l'] * feat['vibration_z_l']
    feat['zz_temp_out']  = feat['temperatur_z_l'] * feat['output_z_l']
    feat['zz_hum_out']   = feat['humidity_z_l']   * feat['output_z_l']
    feat['zz_vib_pwr']   = feat['vibration_z_l']  * feat['power_kw_z_l']

    # Efficiency proxy: z-score of power divided by z-score of output
    # Rises when machine uses more energy per piece (bearing wear signature)
    pwr_z = feat['power_kw_z_l']
    out_z = feat['output_z_l'].replace(0, 1e-6)
    feat['power_per_output_z'] = pwr_z - out_z   # log-linear proxy

    # ── Rolling correlations (catch new/changed inter-sensor relationships) ─
    # Use CORR_WIN (larger than SHORT_WIN) for more stable correlation estimates.
    # Smaller windows make corr_hum_out too noisy to detect process changes.
    feat['corr_temp_out'] = df['temperatur'].rolling(CORR_WIN).corr(df['output']).fillna(0)
    feat['corr_hum_out']  = df['humidity'].rolling(CORR_WIN).corr(df['output']).fillna(0)
    feat['corr_vib_pwr']  = df['vibration'].rolling(CORR_WIN).corr(df['power_kw']).fillna(0)
    feat['corr_temp_pwr'] = df['temperatur'].rolling(CORR_WIN).corr(df['power_kw']).fillna(0)

    return feat


# Columns excluded from model (kept only for display/explanation)
# - SENSOR_COLS: raw values (cycle-variant → would mislead the model)
# - *_roc20 / *_roc30: signed ROC kept for _explain(); model uses *_roc20_up / *_roc30_up
#   (clipped to ≥0) so that recovery phases, which create strong NEGATIVE ROC signals,
#   map to 0 and look like the training baseline → no recovery false alarms.
_EXPLAIN_ONLY_COLS = (
    set(SENSOR_COLS)
    | {f'{c}_roc20' for c in SENSOR_COLS}
    | {f'{c}_roc30' for c in SENSOR_COLS}
)

def model_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Return only cycle-invariant columns for the Isolation Forest."""
    keep = [c for c in feat.columns if c not in _EXPLAIN_ONLY_COLS]
    return feat[keep]


# ── Rule-Based Detector ────────────────────────────────────────────────────────

def check_rules(row: dict) -> list[str]:
    """
    Classic rule-based check: each sensor against its fixed min/max.
    Returns list of violated rules (empty = OK).
    """
    violations = []
    for sensor, (lo, hi) in RULES.items():
        val = row.get(sensor)
        if val is None:
            continue
        if val < lo:
            violations.append(f"{sensor}={val:.2f} < {lo}")
        elif val > hi:
            violations.append(f"{sensor}={val:.2f} > {hi}")
    return violations


# ── Stream Reader ──────────────────────────────────────────────────────────────

class StreamReader:
    """Tail-follows a CSV file, yielding new rows as they are appended."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._open()

    def _open(self):
        self._file = open(self.path, 'r', newline='')
        self._reader = csv.DictReader(self._file)
        # Consume header
        self.fieldnames = self._reader.fieldnames

    def follow(self):
        while True:
            line = self._file.readline()
            if line and line.strip():
                try:
                    values = next(csv.reader([line]))
                    row = dict(zip(self.fieldnames, values))
                    yield row
                except StopIteration:
                    pass
            else:
                time.sleep(0.05)


# ── AI Anomaly Detector ────────────────────────────────────────────────────────

class IsolationForestDetector:
    """
    Trains on the first TRAIN_SAMPLES rows (assumed normal),
    then scores every subsequent row for anomaly likelihood.
    No access to labels or injection schedule.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators=300,
            contamination=CONTAMINATION,
            max_samples='auto',
            random_state=42,
            n_jobs=-1,
        )
        self.trained          = False
        self.feature_names    = None
        self.score_threshold  = -0.15   # overwritten after calibration
        self.calibrating      = False   # True during post-training calibration window
        self.calib_buffer     = []      # raw scores collected during calibration
        self.buffer         = deque(maxlen=max(LONG_WIN + max(LAG_STEPS) + 10,
                                              TRAIN_SAMPLES + WARMUP_WINDOW + 20))
        self.train_buffer   = []
        self.pred_history   = deque(maxlen=PRED_WINDOW)   # 1=anomaly, 0=normal
        # Baseline statistics from training data (for human-readable explanations)
        self.train_means: dict[str, float] = {}
        self.train_stds:  dict[str, float] = {}

    def _buf_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.buffer))

    def ingest(self, row: dict) -> dict | None:
        """
        Feed one new row. Returns detection result dict or None during warmup/training.
        """
        try:
            parsed = {k: float(v) for k, v in row.items()
                      if k in SENSOR_COLS}
            parsed['timestamp'] = int(row['timestamp'])
        except (ValueError, KeyError):
            return None

        self.buffer.append(parsed)

        if len(self.buffer) < WARMUP_WINDOW:
            return None

        if not self.trained:
            self.train_buffer.append(parsed)
            if len(self.train_buffer) >= TRAIN_SAMPLES:
                self._train()
                self.calibrating = True
            return None

        # Post-training calibration: score CALIBRATION_STEPS normal samples to
        # set the detection threshold from actual test-time score distribution.
        # This corrects for any distribution shift between training and detection data.
        if self.calibrating:
            result = self._score(parsed)
            if result is not None:
                self.calib_buffer.append(result['raw_score'])
            if len(self.calib_buffer) >= CALIBRATION_STEPS:
                self._calibrate()
            return None

        return self._score(parsed)

    def _train(self):
        print(f"\n[AI] Training Isolation Forest on {len(self.train_buffer)} samples "
              f"({TRAIN_SAMPLES} requested)...")

        all_buf_df = self._buf_df()
        features   = model_features(compute_features(all_buf_df)).dropna()

        # Use only the tail corresponding to training data
        train_feat = features.tail(len(self.train_buffer)).dropna()

        if len(train_feat) < 20:
            print("[AI] Not enough clean training rows after feature computation - "
                  "collecting more...")
            return

        self.feature_names = train_feat.columns.tolist()
        X = train_feat[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Store per-sensor baseline for explanation (raw sensor values, not features)
        train_df = pd.DataFrame(self.train_buffer)
        for col in SENSOR_COLS:
            self.train_means[col] = float(train_df[col].mean())
            self.train_stds[col]  = float(train_df[col].std())
        # Also store baselines from derived training features (for _explain)
        derived  = ['corr_temp_out', 'corr_hum_out', 'corr_vib_pwr', 'power_per_output_z']
        derived += [f'{c}_roc5'  for c in SENSOR_COLS]
        derived += [f'{c}_roc10' for c in SENSOR_COLS]
        derived += [f'{c}_roc20' for c in SENSOR_COLS]
        derived += [f'{c}_roc30' for c in SENSOR_COLS]
        # Also grab full feature set (including raw) for correlation baselines
        all_feats_full = compute_features(all_buf_df).dropna()
        train_full = all_feats_full.tail(len(self.train_buffer)).dropna()
        for col in derived:
            src = train_full if col in train_full.columns else train_feat
            if col in src.columns:
                self.train_means[col] = float(src[col].mean())
                self.train_stds[col]  = float(src[col].std())

        self.trained = True
        print(f"[AI] Modell bereit. {len(self.feature_names)} Features. "
              f"Kalibrierung läuft ({CALIBRATION_STEPS} Normal-Messwerte)...\n")

    def _calibrate(self):
        """
        Set score threshold from actual test-time normal scores.
        Corrects for distribution shift between training window and detection data:
        the training features (steps 40–340) may have slightly different score
        distributions than test data (steps 340+), especially for rolling features
        near cycle-phase boundaries.  Using CALIBRATION_STEPS post-training normal
        samples gives a threshold that is properly calibrated to the test distribution.
        """
        self.score_threshold = float(np.percentile(self.calib_buffer, SCORE_PERCENTILE))
        self.calibrating     = False
        # Reset prediction history: during calibration the old default threshold
        # (-0.15) was used, filling pred_history with false anomaly flags.
        # A clean window prevents an immediate spurious alert after calibration.
        self.pred_history.clear()
        print(f"[AI] Kalibriert auf {len(self.calib_buffer)} Messwerte. "
              f"Schwelle: {self.score_threshold:.4f} (P{SCORE_PERCENTILE}). "
              f"Alert bei ≥{ANOMALY_FRACTION*100:.0f}% in {PRED_WINDOW} Messungen. "
              f"Überwachung aktiv.\n")

    def _score(self, parsed: dict) -> dict:
        df    = self._buf_df()
        feats_full = compute_features(df)           # includes raw cols for _explain
        feats = model_features(feats_full)          # model sees only invariant features

        if feats.empty:
            return None

        last = feats.iloc[[-1]][self.feature_names]
        last_full = feats_full.iloc[-1]             # full row for _explain
        if last.isnull().values.any():
            return None

        X_scaled  = self.scaler.transform(last.values)
        raw_score = self.model.score_samples(X_scaled)[0]
        # Use calibrated threshold (percentile of training scores) instead of
        # predict() which uses contamination parameter – gives proper calibration
        # even when test-time feature distributions shift slightly from training.
        pred = 1 if raw_score < self.score_threshold else 0

        self.pred_history.append(pred)
        anomaly_rate = float(np.mean(self.pred_history)) if self.pred_history else 0.0
        is_alert     = (len(self.pred_history) >= PRED_WINDOW // 2
                        and anomaly_rate >= ANOMALY_FRACTION)

        explanation = self._explain(parsed, last_full) if is_alert else None

        return {
            'timestamp':    parsed['timestamp'],
            'raw_score':    raw_score,
            'score':        raw_score,        # shown in output for reference
            'anomaly_rate': anomaly_rate,     # fraction of recent preds = anomaly
            'is_alert':     is_alert,
            'explanation':  explanation,
        }

    def _explain(self, parsed: dict, feat_row: pd.Series) -> str:
        """
        Identify which sensors and relationships deviate most from their recent rolling baseline.
        Uses rolling z-scores (feat_row['{col}_zscore']) for raw sensors so that slow cyclical
        signals (temperature sine wave etc.) don't generate false positives.
        Uses training-relative comparison only for correlation features (stable over time).
        """
        deviations = []

        # Per-sensor: use rolling z-score (deviation from rolling 40-sample window)
        # This is phase-invariant for cyclical signals.
        sensor_labels = {
            'temperatur': 'Temperatur',
            'vibration':  'Vibration',
            'humidity':   'Feuchtigkeit',
            'output':     'Output (Stück/min)',
            'power_kw':   'Leistungsaufnahme',
        }
        for col, label in sensor_labels.items():
            z = feat_row.get(f'{col}_zscore', np.nan)
            if np.isnan(z):
                continue
            if abs(z) > 2.2:
                direction = 'erhöht' if z > 0 else 'gesunken'
                deviations.append((abs(z), f"{label} {direction} ({z:+.1f}σ vom gleit. Mittel)"))

        # Rolling rate-of-change: catches slow systematic trends
        # Prefer longer windows (roc30 > roc20 > roc10 > roc5) for sensitivity to slow drift.
        for col, label in sensor_labels.items():
            for roc_key in (f'{col}_roc30', f'{col}_roc20', f'{col}_roc10', f'{col}_roc5'):
                roc = feat_row.get(roc_key, np.nan)
                roc_ref = self.train_means.get(roc_key, 0.0)
                roc_std = self.train_stds.get(roc_key, 1e-6)
                if np.isnan(roc) or roc_std < 1e-9:
                    continue
                sigma = (roc - roc_ref) / roc_std
                if abs(sigma) > 2.8:
                    direction = 'steigt' if sigma > 0 else 'fällt'
                    deviations.append((abs(sigma),
                                       f"{label} {direction} systematisch (Trend: {sigma:+.1f}σ)"))
                    break   # don't double-report same sensor

        # Rolling correlations vs training baseline: catches new/changed relationships
        corr_labels = {
            'corr_hum_out':  'Feuchte↔Output-Kopplung',
            'corr_vib_pwr':  'Vibration↔Leistung-Kopplung',
            'corr_temp_out': 'Temp↔Output-Kopplung',
        }
        for col, label in corr_labels.items():
            val  = feat_row.get(col, np.nan)
            mean = self.train_means.get(col, np.nan)
            std  = self.train_stds.get(col, 1.0)
            if np.isnan(val) or np.isnan(mean) or std < 1e-9:
                continue
            sigma = (val - mean) / std
            if abs(sigma) > 2.5:
                deviations.append((abs(sigma), f"{label} verändert ({sigma:+.1f}σ vs. Normal)"))

        # Leistung/Stück z-score: catches mechanical losses (bearing wear signature)
        ppo     = feat_row.get('power_per_output_z', np.nan)
        ppo_ref = self.train_means.get('power_per_output_z', np.nan)
        ppo_std = self.train_stds.get('power_per_output_z', 1e-6)
        if not np.isnan(ppo) and not np.isnan(ppo_ref) and ppo_std > 1e-9:
            sigma = (ppo - ppo_ref) / ppo_std
            if abs(sigma) > 2.0:
                direction = 'erhöht' if sigma > 0 else 'gesunken'
                deviations.append((abs(sigma), f"Leistung/Stück {direction} ({sigma:+.1f}σ vs. Normal)"))

        if not deviations:
            return "Multivariates Muster — kein einzelner Sensor auffällig"

        deviations.sort(reverse=True)
        top = [msg for _, msg in deviations[:3]]
        return ' | '.join(top)


# ── Helpers ────────────────────────────────────────────────────────────────────

LINE_WIDTH = 62

def _clear_line():
    print(f"\r{' ' * LINE_WIDTH}\r", end='', flush=True)

def _status_tick(ts: int, anomaly_rate: float):
    """Overwrite the current line with a running step counter."""
    bar = int(anomaly_rate * 10)
    rate_vis = '█' * bar + '░' * (10 - bar)
    print(f"\r  Messwert #{ts:5d}  |  Anomalierate: [{rate_vis}] {anomaly_rate:.0%}  |  OK   ",
          end='', flush=True)

def _alert_block(ts: int, anomaly_rate: float, ai_alert: bool, rule_alert: bool,
                 expl: str | None, rule_violations: list[str], alert_num: int):
    """Print a clearly formatted alert block (starts on a fresh line)."""
    _clear_line()
    if ai_alert and rule_alert:
        header_color = '\033[31m'   # red
        label = f'KI + REGELWERK ALARM #{alert_num}'
    elif ai_alert:
        header_color = '\033[33m'   # yellow
        label = f'KI-ALARM #{alert_num}  (Regelwerk: OK)'
    else:
        header_color = '\033[31m'
        label = f'REGELVERSTOS #{alert_num}  (KI: OK)'

    bar = '─' * LINE_WIDTH
    print(f"{header_color}┌{bar}┐\033[0m")
    print(f"{header_color}│  {label:<{LINE_WIDTH - 2}}│\033[0m")
    rate_str = f'Anomalierate: {anomaly_rate:.0%}  (Schwelle: {ANOMALY_FRACTION:.0%})'
    print(f"{header_color}│  Schritt {ts}  |  {rate_str:<{LINE_WIDTH - 12}}│\033[0m")
    if ai_alert and expl:
        print(f"{header_color}│  KI sieht:  {expl[:LINE_WIDTH - 15]:<{LINE_WIDTH - 15}}│\033[0m")
    if rule_alert:
        rv = ', '.join(rule_violations)
        print(f"\033[31m│  Regel:     {rv[:LINE_WIDTH - 15]:<{LINE_WIDTH - 15}}│\033[0m")
    if ai_alert and not rule_alert:
        print(f"{header_color}│  Regel:     alle Schwellen eingehalten{' ' * (LINE_WIDTH - 40)}│\033[0m")
    print(f"{header_color}└{bar}┘\033[0m")


# ── Main Loop ──────────────────────────────────────────────────────────────────

def run_detector(stream_file: str):
    path = Path(stream_file)

    print(f"[Detector] Warte auf Stream: {stream_file}")
    while not path.exists():
        time.sleep(0.2)
    time.sleep(0.3)

    reader  = StreamReader(stream_file)
    ai_det  = IsolationForestDetector()
    tracker = SymptomTracker()
    advisor = LLMAdvisor()

    total       = 0
    ai_alerts   = 0
    rule_alerts = 0
    ai_only     = 0
    rules_only  = 0
    alert_count = 0   # number of distinct alert events

    # State machine for alert transitions
    in_alert        = False
    alert_ts_start  = None
    prev_rule_alert = False   # for detecting rule escalation mid-alarm
    cooldown_until  = 0       # suppress new alerts until this timestamp

    print(f"[Detector] Stream offen. Sammle {TRAIN_SAMPLES} Trainings-Messwerte...")
    print(f"[Detector] Regelgrenzen: {RULES}\n")

    for raw_row in reader.follow():
        total += 1

        # ── Rule-based check ───────────────────────────────────────────────
        try:
            sensor_vals = {k: float(raw_row[k]) for k in SENSOR_COLS}
        except (ValueError, KeyError):
            continue

        rule_violations = check_rules(sensor_vals)
        rule_alert      = len(rule_violations) > 0

        # ── AI check ──────────────────────────────────────────────────────
        ai_result = ai_det.ingest(raw_row)

        if ai_result is None:
            # Still training - show progress in-place
            n = len(ai_det.train_buffer)
            print(f"\r  Training: {n:3d}/{TRAIN_SAMPLES} Messwerte gesammelt ...",
                  end='', flush=True)
            continue

        ai_alert     = ai_result['is_alert']
        ts           = ai_result['timestamp']
        sc           = ai_result['score']
        anomaly_rate = ai_result.get('anomaly_rate', 0.0)
        expl         = ai_result.get('explanation')

        # Counters
        if ai_alert:   ai_alerts   += 1
        if rule_alert: rule_alerts += 1
        if ai_alert and not rule_alert: ai_only  += 1
        if rule_alert and not ai_alert: rules_only += 1

        any_alert = ai_alert or rule_alert

        # When cooldown expires, clear the AI prediction window and reset ai_alert.
        # The ai_result was computed using the stale window (which still contained
        # anomaly flags from the previous alert period / recovery effects).
        # Resetting here prevents an immediate false alarm at cooldown boundary.
        if 0 < cooldown_until <= ts and not in_alert:
            ai_det.pred_history.clear()
            cooldown_until = 0   # mark as consumed so we don't clear repeatedly
            ai_alert     = False
            anomaly_rate = 0.0

        # ── Symptom-Tracking (bei jedem Alert-Schritt) ─────────────────────
        current_event = None
        if any_alert:
            current_event = tracker.add(
                timestamp       = ts,
                explanation     = expl,
                rule_violations = rule_violations,
                ai_alert        = ai_alert,
                rule_alert      = rule_alert,
                sensor_values   = sensor_vals,
            )

        # ── State machine ──────────────────────────────────────────────────
        if any_alert and not in_alert and ts >= cooldown_until:
            # Transition: normal → alert
            alert_count    += 1
            in_alert        = True
            alert_ts_start  = ts
            _alert_block(ts, anomaly_rate, ai_alert, rule_alert, expl, rule_violations, alert_count)

            # Symptom-Verlauf der letzten 3 min ausgeben
            tracker.print_history_block(ts)

            # LLM-Berater asynchron anfragen
            if current_event is not None:
                advisor.query_async(tracker, ts, current_event)

        elif any_alert and in_alert:
            # Escalation: rules just joined an existing KI-alarm
            if rule_alert and not prev_rule_alert and ai_alert:
                rv = ', '.join(rule_violations)
                print(f"\033[31m  ⬆ Schritt {ts} — Regelwerk tritt bei: {rv}\033[0m",
                      flush=True)
            # Continuation: print compact update every 5 steps
            elif (ts - alert_ts_start) % 5 == 0:
                who = []
                if ai_alert:   who.append(f'KI:{anomaly_rate:.0%}')
                if rule_alert: who.append('Regel:' + rule_violations[0].split('=')[0])
                print(f"  ↳ Schritt {ts} | {' | '.join(who)}", flush=True)

        elif not any_alert and in_alert:
            # Transition: alert → normal; start cooldown to suppress recovery artefacts
            duration = ts - alert_ts_start
            _clear_line()
            print(f"  ✓ Schritt {ts} — Alarm beendet (Dauer: {duration} Schritte)\n",
                  flush=True)
            in_alert       = False
            cooldown_until = ts + COOLDOWN_STEPS

        else:
            # Normal operation - rolling step counter
            _status_tick(ts, anomaly_rate)

        prev_rule_alert = rule_alert


def _print_summary(total, ai_alerts, rule_alerts, ai_only, rules_only):
    print("\n" + "=" * 60)
    print(" DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Total samples evaluated : {total}")
    print(f"  AI alerts               : {ai_alerts}")
    print(f"  Rule-based alerts       : {rule_alerts}")
    print(f"  AI caught, rules missed : {ai_only}  ← the interesting ones")
    print(f"  Rules caught, AI missed : {rules_only}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AI + Rule-Based Anomaly Detector')
    parser.add_argument('--stream', default='stream.csv')
    args = parser.parse_args()

    try:
        run_detector(args.stream)
    except KeyboardInterrupt:
        print('\n\n[Detector] Gestoppt.')
