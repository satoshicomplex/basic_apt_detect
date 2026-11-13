"""BTC Complex · SATOSHI COMPLEX · ALL IN ONE FILE (v0 - v6)
Everything in one: billing, v1 - v4 signal detection, v5 reverse-engineer detection,
v6 hardening (cached filters + input guards · exact Decimal money · device-type context
to cut false alarms · a hard 0.9 s answer budget with the full result following).
Run it: python3 satoshi_complex.py  (needs numpy, scipy, scikit-learn)

v5: REVERSE-ENGINEER DETECTION.
A requesting IoT wants to confirm user safety, sub-second; it approves the
waiver and has a payment plan. In under a second satoshi complex:
  1 · triage: is it an emergency?  -> FREE, with a longer hold time on the
      monetizable data until profit sharing pays it back
      otherwise: $5 (USD or $BCH) = 2,000 units · $25 (BTC, BCH, USD or another
      top blockchain) = 12,500 units · or on credit, lowest to no interest
      (1 unit = 1 data unit or 1 neural-network run)
  2 · reverse engineer: are some, most or all of the IoT's own safety / APT
      queries the SAME as the objectives of a software-based attack?
      -> "NO, that is not the case" or "YES" + a score on every point
  3 · at the same time: v4 senses for signals with attack-algorithm patterns,
      finds their direction, ascertains the source, informs the relevant IoTs
  4 · at the same time: requests the IoT's device list or scans to recognize
      ALL the local relevant IoTs, especially smart devices that aren't easy to
      recognize (smart table / desk / bed or other), and where necessary
      establishes communication
  5 · at the same time: requests data or scans for INANIMATE OBJECTS with a
      part of interest to a signal (remote-frequency charging, coils, metal,
      glass, piezo, motors ...) that a sonic / frequency / light / kinetic
      attack could use
  6 · if an attack is detected (signal patterns at the public, OR the IoT's
      own data scoring = attack objectives): gauges the direction + strength
      of the linear or field signal, names the origin / proximity (down to a
      distant transceiver), notifies the IoTs there, requests the source filter
      it, measures any "boosting" (multiples of the level the IoT observes),
      flags severe APTs on public / finance / legal / gov infrastructure,
      sends CISA the required + necessary + obfuscated data (minus what the
      law doesn't allow stored), tells the IoT everything transparently, gives
      a custom strategy, and if the source can't filter in real time, has the
      IoTs search for secondary deployments using different strategies
Sections 1 - 5 below are v0 - v4, each kept whole and loaded as its own module
so every version still runs exactly as tested; section 6 is v5 + v6."""
import sys, types


def _load(name, src):
    """Load one embedded section as its own module (so the versions never collide)."""
    m = types.ModuleType(name); m.__file__ = __file__
    sys.modules[name] = m; exec(compile(src, f"{__file__}:{name}", "exec"), m.__dict__)
    return m


# ==========================================================================
# SECTION 1 · BILLING + ROOM MODEL + SOUND DETECTION (v0, the base · hardened)
# ==========================================================================
_load("integrated", r'''
"""BTC Complex · satoshi complex · ONE FILE: source detection + $BTC / $BCH
data monetization. Each data file = a data unit ($BTC micropayment); each model
run (the anomaly model is the neural-net slot) = a $BCH micropayment.
Part 1 = pure signal processing · Part 2 = model · Part 3 = billing + workflow."""
import hashlib, time
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import butter, sosfilt, stft
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest

# ---- settings -------------------------------------------------------------
FS = 192_000                  # one multichannel interface: mics share a clock
C = 343.0                     # speed of sound, m/s
BAND = (16_000, 45_000)       # audible coil whine + ultrasonic
SLICE = 0.2                   # seconds per slice
DATA_UNIT_BTC_USD = 0.001     # one whole data file = one $BTC micropayment
NEURAL_NET_BCH_USD = 0.0025   # each model / neural net run = one $BCH micropayment
PLAN_USD = 5.00               # data-monetization plan
EMERGENCY_MULTIPLE = 3        # free now; profit share ends up ~3x the plan
DEFENSE_OFFICE = "CISA"       # or another defense office by country / sector

# ---- part 1 · pure signal processing (hardened: cached filters, guards) ----
from scipy.signal import find_peaks
_HP_SOS = butter(2, 1_000, btype="high", fs=FS, output="sos")      # built once at start:
_BP_SOS = butter(6, list(BAND), btype="band", fs=FS, output="sos")  # sub-second every call


def clean(x, lo=BAND[0], hi=BAND[1], fs=FS):
    """Remove DC offset, high-pass, then band-pass (cached filters for the usual band)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1: x = x[:, None]
    x = x - np.mean(x, axis=0)
    if (lo, hi, fs) == (BAND[0], BAND[1], FS):
        hp, bp = _HP_SOS, _BP_SOS
    else:
        hp = butter(2, 1_000, btype="high", fs=fs, output="sos")
        bp = butter(6, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfilt(bp, sosfilt(hp, x, axis=0), axis=0)


def slices(xf, sec=SLICE):
    n = int(sec * FS)
    if n <= 0 or len(xf) < n: return []
    return [xf[s:s + n] for s in range(0, len(xf) - n + 1, n)]


def feature_vector(xs, ref_rms, ref_spec):
    """Per slice: loudness vs room, new tones, pulsing / sweeping, sharp bursts."""
    if xs.size == 0 or ref_spec.size == 0: return np.zeros(4)
    spec = np.abs(np.fft.rfft(xs[:, 0], n=len(ref_spec) * 2 - 2))
    try:
        _, _, Z = stft(xs[:, 0], fs=FS, nperseg=min(2048, len(xs)))
        p = np.max(np.abs(Z), axis=0) if Z.size else np.zeros(1)
    except ValueError:
        p = np.zeros(1)
    k = kurtosis(xs[:, 0])                       # excess kurtosis: 0 for plain noise
    return np.array([
        np.log10(np.sqrt(np.mean(xs ** 2)) / (ref_rms + 1e-12) + 1e-12),
        np.mean(spec > 6 * (ref_spec + 1e-12)),
        np.std(p) / (np.mean(p) + 1e-12),
        0.0 if not np.isfinite(k) else k,
    ])


def gcc_phat_peaks(a, b, max_tau, k=3):
    """Top k arrival-time gaps: 1st = direct path, others = bounces.
    Power-of-2 FFT (fast) · vectorized peak search · never returns empty."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size == 0 or b.size == 0: return [(0.0, 0.0)]
    n = 1 << (len(a) + len(b) - 1).bit_length()
    R = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    cc = np.fft.irfft(R / (np.abs(R) + 1e-12), n)
    m = min(int(np.ceil(FS * max_tau)), n // 2 - 1)
    a_ = np.abs(np.concatenate((cc[-m:], cc[:m + 1]))) if m > 0 else np.abs(cc[:1])
    idx, _ = find_peaks(a_)
    if len(idx) == 0: idx = np.array([int(np.argmax(a_))])     # peak at the edge
    top = idx[np.argsort(-a_[idx])[:k]]
    return [((int(i) - m) / FS, float(a_[i])) for i in top]


def direction_3d(x, mics, max_cond=20.0):
    """3D unit vector + coherence; 0 if the mic geometry can't resolve it."""
    x, mics = np.asarray(x), np.asarray(mics, float)
    if len(mics) < 4 or x.ndim != 2 or x.shape[1] < len(mics):   # 3D needs 4+ mics
        return np.zeros(3), 0.0
    rows, d = [], []
    for i in range(1, len(mics)):
        tau, _ = gcc_phat_peaks(x[:, i], x[:, 0],
                                np.linalg.norm(mics[i] - mics[0]) / C, k=1)[0]
        rows.append(mics[0] - mics[i]); d.append(C * tau)
    A, d = np.array(rows), np.array(d)
    if np.linalg.cond(A) > max_cond:            # mics nearly in a line / plane
        return np.zeros(3), 0.0
    u, *_ = np.linalg.lstsq(A, d, rcond=None)
    nrm = np.linalg.norm(u)
    return u / (nrm + 1e-12), max(0.0, 1 - abs(nrm - 1))   # |u| = 1 if consistent


def triangulate(p1, u1, p2, u2):
    """Two arrays: the meeting point, plus the gap between the lines (quality)."""
    t, *_ = np.linalg.lstsq(np.array([u1, -u2]).T, p2 - p1, rcond=None)
    a, b = p1 + t[0] * u1, p2 + t[1] * u2
    return (a + b) / 2, float(np.linalg.norm(a - b))

# ---- part 2 · anomaly model (the neural-net slot) --------------------------
class RoomModel:
    """Learns the room from quiet captures; flags slices unlike anything normal.
    Isolation forest with 10+ quiet slices; fewer -> a 3-sigma fallback."""
    def __init__(self, quiet_buffers, rate=0.02):
        xs = [clean(b) for b in quiet_buffers if b is not None and len(b)]
        if not xs: raise ValueError("no quiet captures to learn the room from")
        self.rms = float(np.mean([np.sqrt(np.mean(x ** 2)) for x in xs]))
        sl = [s for x in xs for s in slices(x)]
        if not sl: raise ValueError("quiet captures shorter than one slice")
        self.spec = np.median([np.abs(np.fft.rfft(s[:, 0])) for s in sl], axis=0)
        self.rate = rate
        X = np.array([feature_vector(s, self.rms, self.spec) for s in sl])
        self.forest = (IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(X)
                       if len(X) >= 10 else None)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6

    def score(self, xf):                         # xf is already clean()-ed
        sl = slices(xf)
        if not sl: return {"odd_slices": 0, "loud_slices": 0, "anomalous": False}
        X = np.array([feature_vector(s, self.rms, self.spec) for s in sl])
        odd = (self.forest.predict(X) == -1 if self.forest is not None
               else np.any(np.abs((X - self.mu) / self.sd) > 3, axis=1))
        loud = X[:, 0] > np.log10(4)                 # 4x louder than the room
        return {"odd_slices": int(odd.sum()), "loud_slices": int(loud.sum()),
                "anomalous": bool(np.mean(odd & loud) >= 0.3)}

    def update(self, xf):    # only on LOG ONLY windows: room changes slowly
        self.rms += self.rate * (np.sqrt(np.mean(xf ** 2)) - self.rms)

# ---- part 3 · billing + workflow -------------------------------------------
@dataclass
class IoT:
    id: str
    credits_usd: float = 0.0
    plan: str | None = None             # "monetize" or None
    profit_share_owed_usd: float = 0.0  # emergency: paid from future data value
    ledger: list = field(default_factory=list)


@dataclass
class Job:
    iot: IoT
    reason: str
    severity: str                       # "low" | "high" | "emergency"
    mode: str = ""                      # "credits" | "plan" | "emergency"
    data_units: int = 0
    neural_nets: int = 0
    log: list = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def step(self, what):
        self.log.append((round(time.perf_counter() - self.t0, 4), what))


class Settlement:
    """Micropayments are recorded at once, paid in batches: $BTC over Lightning,
    $BCH on chain. Real wallets plug in at pay()."""
    def __init__(self):
        self.pending = {}               # iot id -> {"BTC": usd, "BCH": usd}

    def record(self, iot_id, btc_usd, bch_usd):
        p = self.pending.setdefault(iot_id, {"BTC": 0.0, "BCH": 0.0})
        p["BTC"] += btc_usd; p["BCH"] += bch_usd

    def pay(self, min_usd=1.00):
        out = [(i, cur, round(v, 4)) for i, p in self.pending.items()
               for cur, v in p.items() if v >= min_usd]
        for i, cur, _ in out:
            self.pending[i][cur] = 0.0
        return out                      # -> Lightning invoices / BCH payments


SETTLE = Settlement()


def admit(iot, reason, severity, accepts_waiver, wants_plan=False):
    """Sub-second: reason -> liability waiver -> credits or plan (or emergency)."""
    job = Job(iot, reason, severity)
    job.step(f"request: {reason}")
    if not accepts_waiver:
        job.step("declined: waiver not accepted"); return None, job
    job.step("waiver accepted")
    if iot.credits_usd >= DATA_UNIT_BTC_USD + NEURAL_NET_BCH_USD:
        job.mode = "credits"
    elif wants_plan:
        iot.plan = "monetize"; job.mode = "plan"
    elif severity == "emergency" or iot.credits_usd <= 0:
        job.mode = "emergency"          # free now, profit share later
    else:
        job.step("declined: no credits or plan"); return None, job
    job.step(f"payment mode: {job.mode}")
    return job, job


def handle_data(job, record, monetizable, allowed=True):
    """Store only what can be monetized; the rest is never stored (hash only).
    Whatever is not allowed to be stored is not stored: not even a hash."""
    job.data_units += 1
    if not allowed:
        job.step("data unit not stored (not allowed)"); return {}
    if monetizable:
        job.step("data unit stored (monetizable)"); return {"stored": record}
    job.step("data unit not stored (fingerprint only)")
    return {"fingerprint": hashlib.sha256(repr(record).encode()).hexdigest()[:16]}


def model_run(job, fn, *args):
    """Every model (or neural net) run is counted as $BCH."""
    job.neural_nets += 1
    return fn(*args)


def settle(job, outcome, patched=False):
    """outcome: 'failed' | 'no_attack' | 'attack'."""
    iot = job.iot
    btc = job.data_units * DATA_UNIT_BTC_USD
    bch = job.neural_nets * NEURAL_NET_BCH_USD
    bill = {"btc_units": job.data_units, "bch_nets": job.neural_nets, "usd": 0.0}
    if outcome == "failed":
        job.step("no charge: we failed")
    elif job.mode == "emergency":
        iot.profit_share_owed_usd += EMERGENCY_MULTIPLE * PLAN_USD
        job.step(f"emergency: free now, profit share ~${EMERGENCY_MULTIPLE * PLAN_USD:.2f}")
    else:
        bill["usd"] = round(btc + bch, 6)
        if job.mode == "credits":
            iot.credits_usd = round(iot.credits_usd - btc - bch, 6)
        SETTLE.record(iot.id, btc, bch)
        job.step(f"charged ${btc + bch:.4f} ({outcome})")
    report = None
    if outcome == "attack":
        report = {"to": DEFENSE_OFFICE, "iot": iot.id, "reason": job.reason,
                  "severity": job.severity, "patched": patched,
                  "seconds": round(time.perf_counter() - job.t0, 3)}
        job.step(f"report sent to {DEFENSE_OFFICE}")
    iot.ledger.append(bill)
    return bill, report


def detect_and_bill(job, x, mics, room, can_patch, apply_patch,
                    raw_allowed=False, results_allowed=True):
    """One request: capture -> model -> direction -> fix or forward -> charge."""
    handle_data(job, x, monetizable=True, allowed=raw_allowed)   # raw: voices
    xf = clean(x)
    m = model_run(job, room.score, xf)                         # $BCH: anomaly model
    u, coh = model_run(job, direction_3d, xf, mics)            # $BCH: direction
    why = {**m, "coherence": round(coh, 2)}
    if not (m["anomalous"] and coh > 0.8):
        room.update(xf)
        handle_data(job, {"status": "LOG ONLY", "why": why}, True, results_allowed)
        return {"status": "LOG ONLY", "why": why}, *settle(job, "no_attack")
    timeline = [model_run(job, direction_3d, s, mics)[0].round(2).tolist()
                for s in slices(xf)]                           # $BCH each
    d01 = np.linalg.norm(mics[1] - mics[0])
    result = {"status": "ALERT", "why": why, "direction_xyz": u.round(2).tolist(),
              "timeline_0.2s": timeline,
              "paths_s": gcc_phat_peaks(xf[:, 1], xf[:, 0], d01 / C, k=3)}
    handle_data(job, result, True, results_allowed)
    if job.severity in ("high", "emergency") and can_patch(result):
        apply_patch(result); job.step("patched at the source")
        return result, *settle(job, "attack", patched=True)
    job.step(f"forwarded to {DEFENSE_OFFICE} for a fix")
    return result, *settle(job, "attack", patched=False)
''')


# ==========================================================================
# SECTION 2 · v1 · SOUND + KINETIC + LIGHT + BIOMETRICS
# ==========================================================================
_load("multimodal", r'''
"""BTC Complex · satoshi complex · 2 seconds, in depth: SOUND + KINETIC + LIGHT
+ BIOMETRICS (cortisol, HRV). Only on IoT request. Finds the direction of any
signal with patterns from a severe, undetected or undefended software attack,
cross-references every pattern, and tells ATTACK apart from NOT ATTACK.
Data units = $BTC, model runs = $BCH (billing from integrated.py)."""
import time
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import kurtosis
from integrated import (clean, direction_3d, RoomModel, admit, handle_data,
                        model_run, settle)

FS_K, V_K = 4_000, 300.0      # kinetic: accelerometers on floor / walls, wave m/s
FS_L = 10_000                 # light: 6 photodiodes facing +x -x +y -y +z -z
NORMALS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
AGREE_DEG = 25                # modalities pointing the same way
EVENTS = []                   # anonymous pattern memory: (time, modality, direction)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1))))


def periodicity(env, fs, lo=0.5, hi=30):
    """0..1: how regular the on/off pattern is (engineered pulses are regular)."""
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:] / (np.dot(e, e) + 1e-12)
    a, b = int(fs / hi), int(fs / lo)
    k = a + int(np.argmax(ac[a:b]))
    return float(ac[k]), fs / k                   # strength, pulse rate Hz


def envelope(x, fs, cut=80):
    sos = butter(2, cut, fs=fs, output="sos")
    return sosfilt(sos, np.abs(x))[:: max(1, fs // 400)]   # ~400 Hz envelope


def tdoa_direction(x, pos, v, fs):
    """Plane-wave direction from arrival-time gaps (any wave speed)."""
    rows, d = [], []
    for i in range(1, len(pos)):
        n = 2 * len(x)
        R = np.fft.rfft(x[:, i], n) * np.conj(np.fft.rfft(x[:, 0], n))
        cc = np.fft.irfft(R / (np.abs(R) + 1e-12), n)
        m = int(np.ceil(fs * np.linalg.norm(pos[i] - pos[0]) / v)) + 1
        cc = np.concatenate((cc[-m:], cc[:m + 1]))
        rows.append(pos[0] - pos[i]); d.append(v * (np.argmax(cc) - m) / fs)
    u, *_ = np.linalg.lstsq(np.array(rows), np.array(d), rcond=None)
    return unit(u), max(0.0, 1 - abs(np.linalg.norm(u) - 1))


class Baseline:
    """Per modality: what this room is normally like (learned on quiet captures)."""
    def __init__(self, quiet, fs, band):
        self.fs, self.band = fs, band
        f = [self.feats(q) for q in quiet]
        self.rms = np.mean([x["rms"] for x in f])
        self.spec = np.median([x["spec"] for x in f], axis=0)
        self.ch = np.mean([np.sqrt(np.mean(x["xf"] ** 2, axis=0)) for x in f], axis=0)

    def filt(self, x):
        sos = butter(4, self.band, btype="band", fs=self.fs, output="sos")
        return sosfilt(sos, x - x.mean(axis=0), axis=0)

    def feats(self, x):
        xf = self.filt(x)
        s = np.abs(np.fft.rfft(xf[:, 0]))
        return {"xf": xf, "rms": np.sqrt(np.mean(xf ** 2)), "spec": s / (s.sum() + 1e-12)}


def kinetic(x, pos, base):
    """Vibration through floor / walls / structure: is it new, engineered, from where?"""
    f = base.feats(x)
    loud = f["rms"] / (base.rms + 1e-12)
    new = float(np.sum(np.clip(f["spec"] - 3 * base.spec, 0, None)))
    reg, rate = periodicity(envelope(f["xf"][:, 0], FS_K), 400)
    u, coh = tdoa_direction(f["xf"], pos, V_K, FS_K)
    return {"mod": "kinetic", "anomalous": loud > 1.2 and new > 0.3,
            "engineered": reg > 0.6 or kurtosis(f["xf"][:, 0]) > 3,
            "u": u, "coh": coh, "detail": f"{loud:.0f}x room, pulses {rate:.1f}/s ({reg:.2f})"}


def light(x, base):
    """Light patterns: flicker / strobe / coded pulses beyond the room's normal lights."""
    dc = x.mean(axis=0) + 1e-9
    f = base.feats(x)
    depth = float(np.max(np.sqrt(np.clip(np.mean(f["xf"] ** 2, axis=0)
                                         - base.ch ** 2, 0, None)) / dc))
    new = float(np.sum(np.clip(f["spec"] - 3 * base.spec, 0, None)))
    reg, rate = periodicity(envelope(f["xf"][:, int(np.argmax(np.std(f["xf"], 0)))], FS_L), 400)
    ac = np.sqrt(np.clip(np.mean(f["xf"] ** 2, axis=0) - base.ch ** 2, 0, None))
    v = ac @ NORMALS                           # brighter-flicker side = direction
    return {"mod": "light", "anomalous": depth > 0.05 and new > 0.3,
            "engineered": reg > 0.6, "u": unit(v),
            "coh": float(np.dot(v, v) / (np.dot(ac, ac) + 1e-12)),  # 1 = one source
            "detail": f"modulation {100 * depth:.0f}%, {rate:.1f} Hz pattern ({reg:.2f})"}


def sound(x, mics, room):
    xf = clean(x)
    m = room.score(xf)
    u, coh = direction_3d(xf, mics)
    return {"mod": "sound", "anomalous": m["anomalous"], "engineered": m["anomalous"],
            "u": u, "coh": coh, "detail": f"{m['odd_slices']} odd slices"}


def biometrics(b):
    """Cross-reference only. HRV / heart rate react in seconds; cortisol rises
    ~10-40 min after a stressor, so it confirms REPEATED exposures, not one."""
    now = b["hrv_ms"] < 0.75 * b["hrv_base"] or b["hr"] > 1.15 * b["hr_base"]
    z = (b["cortisol"][-1][1] - b["cort_base"]) / (b["cort_sd"] + 1e-9)
    return {"stress_now": bool(now), "cortisol_z": round(float(z), 1)}


def cross_reference(ev, bio, benign, now):
    """Vital: tell ATTACK apart from NOT ATTACK, with the reasons."""
    why_not, live = [], []
    for e in ev:
        if not e["anomalous"]:
            continue
        if e["coh"] < 0.6:
            why_not.append(f"{e['mod']}: diffuse, no single direction"); continue
        k = next((n for n, (m, d) in benign.items()
                  if m == e["mod"] and angle(d, e["u"]) < AGREE_DEG), None)
        if k:
            why_not.append(f"{e['mod']}: matches known device '{k}'"); continue
        if not e["engineered"]:
            why_not.append(f"{e['mod']}: unusual but natural / irregular"); continue
        live.append(e)
    score = len(live)
    agree = [e for e in live if any(angle(e["u"], o["u"]) < AGREE_DEG
                                    for o in live if o is not e)]
    if len(agree) >= 2: score += 1.5                          # same direction
    u = unit(sum(e["u"] * e["coh"] for e in (agree or live))) if live else None
    past = [p for p in EVENTS if u is not None and angle(p[2], u) < AGREE_DEG
            and now - p[0] < 7 * 86400]
    if past: score += 1                                       # repeats from there
    if live and bio["stress_now"]: score += 0.5
    recent = [p for p in past if 600 <= now - p[0] <= 2400]   # 10-40 min ago
    if live and recent and bio["cortisol_z"] >= 2: score += 1  # cortisol followed
    for e in live: EVENTS.append((now, e["mod"], e["u"]))
    status = ("ATTACK" if score >= 3 else "POSSIBLE ATTACK" if score >= 1.5
              else "NOT ATTACK")
    return {"status": status, "score": score, "direction_xyz":
            None if u is None else u.round(2).tolist(),
            "signals": [f"{e['mod']}: {e['detail']}" for e in live],
            "agree": [e["mod"] for e in agree], "repeats_7d": len(past),
            "biometrics": bio, "why_not_attack": why_not}


def analyze_2s(job, cap, rig, benign, now=None):
    """cap: 2 s of sound (192 kHz), kinetic (4 kHz), light (10 kHz) + biometrics."""
    now = time.time() if now is None else now
    handle_data(job, cap["sound"], monetizable=True, allowed=False)   # voices
    handle_data(job, cap["kinetic"], monetizable=True, allowed=True)
    handle_data(job, cap["light"], monetizable=True, allowed=True)
    handle_data(job, cap["bio"], monetizable=False, allowed=False)    # health
    ev = [model_run(job, sound, cap["sound"], rig["mics"], rig["room"]),
          model_run(job, kinetic, cap["kinetic"], rig["accel"], rig["kin_base"]),
          model_run(job, light, cap["light"], rig["light_base"])]
    bio = model_run(job, biometrics, cap["bio"])
    out = model_run(job, cross_reference, ev, bio, benign, now)
    handle_data(job, {k: v for k, v in out.items() if k != "biometrics"}, True, True)
    if out["status"] == "NOT ATTACK":
        return out, *settle(job, "no_attack")
    return out, *settle(job, "attack")       # report to CISA with the direction
''')


# ==========================================================================
# SECTION 3 · v2 · 0.4 s WINDOWS · MASKED-PHAT DIRECTION · FUSION · SIGNED REGISTRY
# ==========================================================================
_load("multimodal_v2", r'''
"""BTC Complex · satoshi complex · v2: SUB-SECOND, STREAMING, SOUND + KINETIC +
LIGHT + BIOMETRICS. Builds on v1 with the best of Grok's review: 0.4 s windows
every 0.2 s (first answer after 0.4 s of data), early exit on quiet windows,
adaptive baselines, wider "engineered" check (pulses, sweeps / chirps, coded
on-off, sharp bursts), direction held over time, local vs distant hint,
weighted confidence, bounded memory, signed benign list + signed evidence.
Data units = $BTC, model runs = $BCH (billing from integrated.py)."""
import hashlib, hmac, json, time
from collections import deque
import numpy as np
from scipy.stats import kurtosis
from integrated import clean, admit, handle_data, model_run, settle
from multimodal import (FS_K, V_K, FS_L, NORMALS, unit, angle, periodicity,
                        envelope, tdoa_direction, Baseline, biometrics)
import integrated as I

WIN, HOP = 0.4, 0.2           # seconds: sub-second reaction
AGREE_DEG = 20.0
EVENTS = deque(maxlen=5000)   # bounded anonymous memory: (time, mod, u, fingerprint)
KEY = b"replace-with-a-device-key"   # signs the benign list + evidence packages


def sign(obj):
    return hmac.new(KEY, json.dumps(obj, sort_keys=True, default=str).encode(),
                    hashlib.sha256).hexdigest()


def bspec(base, n):
    """Room spectrum resampled to this window's length."""
    s = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(base.spec)), base.spec)
    return s / (s.sum() + 1e-12)


def bspec_to(spec, n):
    s = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(spec)), spec)
    return s / (s.sum() + 1e-12)


def adapt(base, f, rate=0.01):
    """Adaptive baseline: on clear windows the room slowly updates itself."""
    base.rms += rate * (f["rms"] - base.rms)
    base.spec += rate * (bspec_to(f["spec"], len(base.spec)) - base.spec)
    base.ch += rate * (np.sqrt(np.mean(f["xf"] ** 2, axis=0)) - base.ch)


def dominant(x, fs):
    s = np.abs(np.fft.rfft(x)); return float(np.argmax(s[1:]) + 1) * fs / len(x)


def engineered(x, fs):
    """0..1 plus why: regular pulses · sweep / chirp · coded on-off · sharp bursts."""
    env = np.clip(envelope(x, fs), 0, None)
    reg, rate = periodicity(env, 400) if len(env) > 60 else (0.0, 0.0)
    parts = np.array_split(x, 8)
    f = np.array([dominant(p, fs) for p in parts])
    sweep = abs(np.corrcoef(np.arange(8), f)[0, 1]) if f.std() > 0.01 * f.mean() else 0.0
    e = env / (env.sum() + 1e-12)
    ent = -np.sum(e * np.log(e + 1e-12)) / np.log(len(e))     # 1 = steady, low = on/off
    k = kurtosis(x)
    s = {"pulses": reg, "sweep": float(np.nan_to_num(sweep)),
         "coded on-off": float(np.clip((0.97 - ent) * 10, 0, 1)), "bursts": min(1.0, k / 6)}
    top = max(s, key=s.get)
    return float(min(1.0, max(s.values()))), f"{top} {s[top]:.2f}" + (
        f", {rate:.1f}/s" if top == "pulses" else "")


def direction(x, pos, v, fs, max_cond=20.0):
    """Arrival-time direction with masked PHAT: whiten only the frequencies the
    signal really uses, so short or narrow-band bursts aren't lost in noise."""
    rows, d = [], []
    n = 2 * len(x)
    X0 = np.fft.rfft(x[:, 0], n)
    for i in range(1, len(pos)):
        R = np.fft.rfft(x[:, i], n) * np.conj(X0)
        a = np.abs(R); w = (a > 0.05 * a.max()) / (a + 1e-12)
        cc = np.fft.irfft(R * w, n)
        m = int(np.ceil(fs * np.linalg.norm(pos[i] - pos[0]) / v)) + 1
        cc = np.concatenate((cc[-m:], cc[:m + 1]))
        rows.append(pos[0] - pos[i]); d.append(v * (np.argmax(cc) - m) / fs)
    A = np.array(rows)
    if np.linalg.cond(A) > max_cond: return np.zeros(3), 0.0
    u, *_ = np.linalg.lstsq(A, np.array(d), rcond=None)
    return unit(u), max(0.0, 1 - abs(np.linalg.norm(u) - 1))


def burst_mask(x, fs, block=0.005):
    """Keep only the loud parts, so short chirps / bursts aren't lost in noise."""
    n = int(block * fs); m = len(x) // n
    r = np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1))
    keep = np.repeat(r > 0.25 * r.max(), n)
    return np.concatenate([keep, np.zeros(len(x) - m * n, bool)]).astype(float)


def range_hint(per_sensor_rms):
    """Local vs distant: a near source is much stronger at the closest sensor."""
    g = float(np.max(per_sensor_rms) / (np.min(per_sensor_rms) + 1e-12))
    return ("local" if g > 1.6 else "distant"), round(g, 2)


def look(mod, xw, rig):
    """One modality, one window: new? engineered? direction? local / distant?"""
    if mod == "sound":
        xf = clean(xw); m = rig["room"].score(xf)
        keep = burst_mask(xf[:, 0], I.FS) > 0                   # loud parts only
        u, coh = direction(xf[keep] if keep.sum() > 256 else xf, rig["mics"], I.C, I.FS)
        eng, why = engineered(xf[:, 0], I.FS)
        rh = range_hint(np.sqrt(np.mean(xf ** 2, axis=0)))
        return {"mod": mod, "anomalous": m["anomalous"], "eng": eng, "why": why,
                "u": u, "coh": coh, "range": rh, "freq": dominant(xf[:, 0], I.FS)}
    base = rig["kin_base"] if mod == "kinetic" else rig["light_base"]
    fs = FS_K if mod == "kinetic" else FS_L
    f = base.feats(xw)
    ac = np.sqrt(np.clip(np.mean(f["xf"] ** 2, axis=0) - base.ch ** 2, 0, None))
    new = float(np.sum(np.clip(f["spec"] - 3 * bspec(base, len(f["spec"])), 0, None)))
    if mod == "kinetic":
        loud = f["rms"] / (base.rms + 1e-12)
        anomalous = new > 0.3 and loud > 0.8   # new energy, not just louder
        u, coh = tdoa_direction(f["xf"], rig["accel"], V_K, FS_K)
        ch = int(np.argmax(ac)); rh = range_hint(np.sqrt(np.mean(f["xf"] ** 2, axis=0)))
    else:
        depth = float(np.max(ac / (xw.mean(axis=0) + 1e-9)))
        anomalous = depth > 0.04 and new > 0.25
        v = ac @ NORMALS; u = unit(v); coh = float(v @ v / (ac @ ac + 1e-12))
        ch = int(np.argmax(ac)); rh = ("n/a", 0.0)        # 6 sensors: direction only
    eng, why = engineered(f["xf"][:, ch], fs)
    return {"mod": mod, "anomalous": anomalous, "eng": eng, "why": why, "u": u,
            "coh": coh, "range": rh, "_f": f,
            "freq": float(np.argmax(np.clip(f["spec"] - 3 * bspec(base, len(f["spec"])), 0, None)))
                    * fs / (2 * (len(f["spec"]) - 1))}            # the NEW frequency


def is_benign(e, registry):
    """Signed list: device = modality + direction + frequency fingerprint."""
    for name, d in registry["devices"].items():
        if (d["mod"] == e["mod"] and angle(d["u"], e["u"]) < AGREE_DEG
                and abs(d["freq"] - e["freq"]) <= 0.1 * d["freq"]):
            return name
    return None


def fuse(windows, bio, registry, now):
    """Weighted evidence -> confidence. Biometrics only add to a physical signal."""
    assert hmac.compare_digest(registry["sig"], sign(registry["devices"])), "benign list tampered"
    why_not, live = set(), []
    for t, ev in windows:
        for e in ev:
            if not e["anomalous"]: continue
            if e["coh"] < 0.6: why_not.add(f"{e['mod']}: diffuse, no single direction"); continue
            k = is_benign(e, registry)
            if k: why_not.add(f"{e['mod']}: matches signed known device '{k}'"); continue
            if e["eng"] < 0.5: why_not.add(f"{e['mod']}: unusual but natural / irregular"); continue
            live.append((t, e))
    if not live:
        return {"status": "NOT ATTACK", "confidence": 0.0, "first_alert_s": None,
                "direction_xyz": None, "why_not_attack": sorted(why_not), "action": "clear"}
    u = unit(sum(e["u"] * e["coh"] * e["eng"] for _, e in live))
    held = [(t, e) for t, e in live if angle(e["u"], u) < AGREE_DEG]
    if not held:                              # sources disagree: follow the strongest
        u = max(live, key=lambda q: q[1]["coh"] * q[1]["eng"])[1]["u"]
        held = [(t, e) for t, e in live if angle(e["u"], u) < AGREE_DEG]
    mods = sorted({e["mod"] for _, e in held})
    active = sum(1 for _, ev in windows if ev) or 1
    persist = len({t for t, _ in held}) / active        # share of non-quiet windows
    fp = hashlib.sha256(json.dumps(sorted((e["mod"], round(e["freq"], -1)) for _, e in held))
                        .encode()).hexdigest()[:12]                # pattern fingerprint
    past = [p for p in EVENTS if angle(p[2], u) < AGREE_DEG and now - p[0] < 7 * 86400]
    same_fp = sum(p[3] == fp for p in past)
    recent = [p for p in past if 600 <= now - p[0] <= 2400]
    z = (0.9 * len(mods) + 1.2 * (len(mods) >= 2) + 1.5 * persist
         + 0.8 * min(1, len(past)) + 0.6 * min(1, same_fp)
         + 0.4 * bio["stress_now"] + 0.8 * (bool(recent) and bio["cortisol_z"] >= 2)
         + np.mean([e["eng"] for _, e in held]) - 2.6)
    conf = float(1 / (1 + np.exp(-2 * z)))
    for m in mods: EVENTS.append((now, m, u, fp))
    status = "ATTACK" if conf >= 0.8 else "POSSIBLE ATTACK" if conf >= 0.5 else "NOT ATTACK"
    rng = [e["range"] for _, e in held if e["range"][0] != "n/a"]
    return {"status": status, "confidence": round(conf, 2),
            "first_alert_s": round(min(t for t, _ in held) + WIN, 1),
            "direction_xyz": u.round(2).tolist(), "modalities": mods,
            "held_over_windows": f"{100 * persist:.0f}%",
            "range": max(rng, key=lambda r: r[1])[0] if rng else "unknown",
            "pattern": [f"{m}: " + max((e for _, e in held if e["mod"] == m),
                                        key=lambda e: e["eng"])["why"] for m in mods],
            "fingerprint": fp, "repeats_7d": len(past), "biometrics": bio,
            "why_not_attack": sorted(why_not),
            "action": {"ATTACK": "patch locally if possible, else escalate signed to CISA",
                       "POSSIBLE ATTACK": "monitor + isolate the nearest device that way",
                       "NOT ATTACK": "clear"}[status]}


def is_quiet(cut, rig):
    """Cheap check: nothing louder than the room in sound, kinetic or light."""
    return (np.std(clean(cut["sound"])) < 2 * rig["room"].rms
            and np.std(rig["kin_base"].filt(cut["kinetic"])) < 1.3 * rig["kin_base"].rms
            and np.std(cut["light"] - cut["light"].mean(0)) < 2 * rig["light_base"].ch.max())


def analyze_stream(job, cap, rig, registry, now=None):
    """2 s (or a live ring buffer) in 0.4 s windows every 0.2 s; quiet windows exit early."""
    now = time.time() if now is None else now
    t_start = time.perf_counter()
    handle_data(job, cap["sound"], monetizable=True, allowed=False)   # voices
    handle_data(job, cap["kinetic"], monetizable=True, allowed=True)
    handle_data(job, cap["light"], monetizable=True, allowed=True)
    handle_data(job, cap["bio"], monetizable=False, allowed=False)    # health
    fs = {"sound": I.FS, "kinetic": FS_K, "light": FS_L}
    windows, quiet = [], 0
    for t in np.arange(0, 2.0 - WIN + 1e-9, HOP):
        cut = {m: cap[m][int(t * fs[m]): int((t + WIN) * fs[m])] for m in fs}
        if is_quiet(cut, rig):                                         # early exit:
            quiet += 1; windows.append((t, [])); continue                 # no $BCH
        windows.append((t, [model_run(job, look, m, cut[m], rig) for m in fs]))
    bio = model_run(job, biometrics, cap["bio"])
    out = model_run(job, fuse, windows, bio, registry, now)
    out["quiet_windows_skipped"] = quiet
    out["ms"] = round(1000 * (time.perf_counter() - t_start))
    if out["status"] == "NOT ATTACK":                                  # room adapts
        for _, ev in windows:
            for e in ev:
                if "_f" in e: adapt(rig[f"{'kin' if e['mod'] == 'kinetic' else 'light'}_base"], e["_f"])
    evidence = {k: v for k, v in out.items() if k != "biometrics"}
    out["evidence_sig"] = sign(evidence)                             # signed package
    handle_data(job, evidence, True, True)
    return out, *settle(job, "no_attack" if out["status"] == "NOT ATTACK" else "attack")
''')


# ==========================================================================
# SECTION 4 · v3 · DEVICE KEYS · RIG CHECK · DISTANCE · CONTAIN · STIX 2.1 TO CISA
# ==========================================================================
_load("multimodal_v3", r'''
"""BTC Complex · satoshi complex · v3 DRAFT: Google + Grok + Claude, to make the
App strong. Keeps v2's sub-second streaming detection (sound, kinetic, light,
biometrics) and adds:
  security  · key from TPM / secure element / env, never in the code
            · canonical JSON before signing · per-device memory + baselines
            · benign list signed AND hash-chained (no silent inserts)
  physics   · rig check: synchronized arrays of 4+, clock sync, sample rates;
              a modality that fails gets intensity only, no direction
            · kinetic wave speed calibrated per building · local vs distant
              from wavefront curvature (distance up to ~0.6 m), not just loudness
            · kinetic + light judged only on what is NEW vs the room
  tested    · simulated validation: share of attacks caught vs false alarms
  response  · containment first, sub-second (isolate, mute, close aperture,
              alert); firmware patch + forensics on a slower, signed path
            · STIX 2.1 report for CISA / a SOC over authenticated TAXII
            · billing ($BTC / $BCH) settles in the background, never blocks
Data units = $BTC, model runs = $BCH (billing from integrated.py)."""
import hashlib, hmac, json, os, queue, threading, time, uuid
from collections import deque
import numpy as np
import multimodal_v2 as V
from multimodal_v2 import unit, angle
from integrated import admit, settle, clean, C, FS as FS_S

NEED = {"sound": (4, 96_000, 10e-6), "kinetic": (4, 1_000, 100e-6),   # sensors,
        "light": (6, 2_000, 1e-3)}                                   # Hz, sync s


def load_key(dev_id):
    """TPM / secure element first; env var for tests; never a key in the code."""
    k = os.environ.get(f"BTCC_KEY_{dev_id}") or os.environ.get("BTCC_DEVICE_KEY")
    if not k:
        raise RuntimeError("no device key: provision it in the TPM / secure element")
    return k.encode()


def canonical(o):
    """Same bytes on every machine: plain numbers, rounded, sorted, no spaces."""
    if isinstance(o, dict): return {str(k): canonical(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple, np.ndarray)): return [canonical(v) for v in list(o)]
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (int, np.integer)): return int(o)
    if isinstance(o, (float, np.floating)): return round(float(o), 6)
    return o if o is None or isinstance(o, str) else str(o)


class Device:
    """Everything per device / sensor array: key, memory, baselines, registry."""
    _lock = threading.Lock()

    def __init__(self, dev_id, rig, sync_s, calibration):
        self.id, self.rig, self.sync_s = dev_id, rig, sync_s
        self.key = load_key(dev_id)
        self.events = deque(maxlen=5000)
        self.registry = []                              # hash-chained entries
        self.cal_hash = hashlib.sha256(json.dumps(canonical(calibration)).encode()).hexdigest()
        self.directional = self.check_rig()

    def sign(self, obj):
        b = json.dumps(canonical(obj), sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self.key, b, hashlib.sha256).hexdigest()

    def check_rig(self):
        """Direction only where the physics allows it."""
        have = {"sound": (len(self.rig["mics"]), FS_S), "kinetic": (len(self.rig["accel"]), V.FS_K),
                "light": (len(V.NORMALS), V.FS_L)}
        return {m: have[m][0] >= NEED[m][0] and have[m][1] >= NEED[m][1]
                and self.sync_s.get(m, 1) <= NEED[m][2] for m in NEED}

    # -- benign list: each entry signed + chained to the one before -----------
    def add_benign(self, name, mod, u, freq):
        prev = self.registry[-1]["hash"] if self.registry else "genesis"
        e = {"name": name, "mod": mod, "u": canonical(u), "freq": float(freq), "prev": prev}
        e["hash"] = hashlib.sha256(json.dumps(canonical(e), sort_keys=True).encode()).hexdigest()
        e["sig"] = self.sign(e["hash"]); self.registry.append(e)

    def registry_ok(self):
        prev = "genesis"
        for e in self.registry:
            body = {k: e[k] for k in ("name", "mod", "u", "freq", "prev")}
            h = hashlib.sha256(json.dumps(canonical(body), sort_keys=True).encode()).hexdigest()
            if e["prev"] != prev or h != e["hash"] or not hmac.compare_digest(e["sig"], self.sign(h)):
                return False
            prev = e["hash"]
        return True


def calibrate_speed(x, pos, src, fs, speeds=np.arange(100, 3001, 25)):
    """Per building: tap at a known spot, keep the wave speed that fits best."""
    xf = x - x.mean(0); tau = tdoas(xf, pos, 3000.0, fs)
    best = min(speeds, key=lambda v: np.sum((np.array(
        [(np.linalg.norm(src - p) - np.linalg.norm(src - pos[0])) / v for p in pos[1:]]) - tau) ** 2))
    return float(best)


def tdoas(x, pos, vmax, fs):
    """Arrival-time gaps vs sensor 0 (masked PHAT, sub-sample peak)."""
    out, n = [], 2 * len(x)
    X0 = np.fft.rfft(x[:, 0], n)
    for i in range(1, len(pos)):
        R = np.fft.rfft(x[:, i], n) * np.conj(X0)
        a = np.abs(R); cc = np.fft.irfft(R * (a > 0.05 * a.max()) / (a + 1e-12), n)
        m = int(np.ceil(fs * np.linalg.norm(pos[i] - pos[0]) / 100.0)) + 2
        m = min(m, n // 2 - 2); cc = np.concatenate((cc[-m:], cc[:m + 1]))
        k = int(np.argmax(cc)); y0, y1, y2 = cc[max(k - 1, 0)], cc[k], cc[min(k + 1, len(cc) - 1)]
        frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        out.append((k - m + frac) / fs)
    return np.array(out)


def distance(xf, mics, u):
    """Wavefront curvature: fit the source distance along u (near = curved)."""
    keep = V.burst_mask(xf[:, 0], FS_S) > 0
    tau = tdoas(xf[keep] if keep.sum() > 256 else xf, mics, C, FS_S)
    from scipy.optimize import least_squares
    c0 = mics.mean(0)
    f = lambda s: (np.linalg.norm(s - mics[1:], axis=1) - np.linalg.norm(s - mics[0])) / C - tau
    fits = [least_squares(f, c0 + r * u, x_scale=1.0) for r in (0.2, 0.5, 1, 3, 10)]
    best = min(fits, key=lambda q: q.cost)
    r = float(min(np.linalg.norm(best.x - c0), 50.0))
    # a 15 cm array can tell curvature only up to ~4x its size: beyond that, just "not local"
    return (round(r, 2), "local" if r <= 0.6 else "not local (beyond ~0.6 m)")


LOOK_V2, QUIET_V2 = V.look, V.is_quiet


def smooth(s, k=9):
    """Spectrum averaged over 9 bins, so random noise peaks don't look 'new'."""
    s = np.convolve(s, np.ones(k) / k, "same"); return s / (s.sum() + 1e-12)


def quiet3(cut, rig):
    """v3 gate: skip only if nothing is LOUDER and nothing is NEW (one cheap FFT
    per sense), so a weak attack hiding under the fridge isn't skipped."""
    if not QUIET_V2(cut, rig): return False
    for m, base in (("kinetic", rig["kin_base"]), ("light", rig["light_base"])):
        x = base.filt(cut[m]); s = smooth(np.abs(np.fft.rfft(x[:, int(np.argmax(x.std(0)))])))
        if np.sum(s * (s > 3 * smooth(V.bspec(base, len(s))))) > 0.08: return False
    return True


def look3(mod, xw, rig):
    """v3: for kinetic + light, look only at what is NEW vs the room: keep the
    frequencies the room doesn't have, rebuild that part, then judge it."""
    if mod == "sound":
        return LOOK_V2(mod, xw, rig)
    base = rig["kin_base"] if mod == "kinetic" else rig["light_base"]
    fs = V.FS_K if mod == "kinetic" else V.FS_L
    f = base.feats(xw)
    X = np.fft.rfft(f["xf"], axis=0)
    c0 = int(np.argmax(np.sum(np.abs(X) ** 2, axis=0)))         # the sensor seeing most
    sp = smooth(np.abs(X[:, c0]))
    mask = sp > 3 * smooth(V.bspec(base, len(sp)))              # new frequencies
    p = np.abs(X[:, c0]) ** 2
    new = float(np.sum(p * mask) / (p.sum() + 1e-12))
    xr = np.fft.irfft(X * mask[:, None], n=len(f["xf"]), axis=0)   # only the new part
    ac = np.sqrt(np.mean(xr ** 2, axis=0)); ch = int(np.argmax(ac))
    if mod == "kinetic":
        anomalous = new > 0.08 and ac.max() > 0.15 * base.rms
        u, coh = V.tdoa_direction(xr, rig["accel"], V.V_K, V.FS_K)
        if coh < 0.8:            # narrow tone: the carrier repeats, so use its on/off edges
            from scipy.signal import butter, sosfilt
            env = sosfilt(butter(2, 200, fs=V.FS_K, output="sos"), np.abs(xr), axis=0)
            u2, c2 = V.tdoa_direction(env - env.mean(0), rig["accel"], V.V_K, V.FS_K)
            if c2 > coh: u, coh = u2, c2
        rh = V.range_hint(ac)
    else:
        anomalous = new > 0.08 and float(np.max(ac / (xw.mean(axis=0) + 1e-9))) > 0.02
        v = ac @ V.NORMALS; u = unit(v); coh = float(v @ v / (ac @ ac + 1e-12))
        rh = ("n/a", 0.0)
    eng, why = V.engineered(xr[:, ch], fs)
    freq = float(np.argmax(p * mask)) * fs / len(f["xf"])
    return {"mod": mod, "anomalous": anomalous, "eng": eng, "why": why, "u": u,
            "coh": coh, "range": rh, "_f": f, "freq": freq}


CONTAIN = {"sound": "mute the nearest speaker / audio inputs",
           "kinetic": "stop the nearest motor / actuator",
           "light": "close blinds / aperture, switch off the nearest smart light"}


def contain(out, do):
    """Sub-second: contain first. Patching is NOT sub-second: it is queued."""
    if out["status"] == "NOT ATTACK": return [], []
    now = ["alert user + IoT", "isolate the nearest networked device (VLAN / block)"]
    now += [CONTAIN[m] for m in out.get("modalities", [])]
    if out["status"] == "POSSIBLE ATTACK": now = now[:1] + ["watch closely, 2 s in depth"]
    for a in now: do(a)
    later = (["signed firmware / config fix", "full forensic capture"]
             if out["status"] == "ATTACK" else [])
    return now, later


def stix(dev, out, evidence_sha):
    """STIX 2.1 bundle: identity + observed data + indicator + sighting."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    ident = {"type": "identity", "spec_version": "2.1", "id": f"identity--{uuid.uuid4()}",
             "name": f"btcc-device-{dev.id}", "identity_class": "system", "created": ts, "modified": ts}
    obs = {"type": "observed-data", "spec_version": "2.1", "id": f"observed-data--{uuid.uuid4()}",
           "created": ts, "modified": ts, "first_observed": ts, "last_observed": ts,
           "number_observed": 1, "object_refs": [ident["id"]],
           "x_btcc": {"direction_xyz": out["direction_xyz"], "range": out.get("range"),
                      "modalities": out.get("modalities"), "pattern": out.get("pattern"),
                      "fingerprint": out.get("fingerprint"), "time_uncertainty_s": 0.2,
                      "calibration_sha256": dev.cal_hash, "evidence_sha256": evidence_sha}}
    ind = {"type": "indicator", "spec_version": "2.1", "id": f"indicator--{uuid.uuid4()}",
           "created": ts, "modified": ts, "valid_from": ts, "pattern_type": "stix",
           "pattern": f"[x-btcc-signal:fingerprint = '{out.get('fingerprint')}']",
           "confidence": int(100 * out["confidence"]), "name": "physical-channel attack pattern"}
    sig = {"type": "sighting", "spec_version": "2.1", "id": f"sighting--{uuid.uuid4()}",
           "created": ts, "modified": ts, "sighting_of_ref": ind["id"],
           "observed_data_refs": [obs["id"]], "where_sighted_refs": [ident["id"]]}
    return {"type": "bundle", "id": f"bundle--{uuid.uuid4()}", "objects": [ident, obs, ind, sig]}


def send_taxii(bundle, url, token):
    """Authenticated TAXII 2.1 POST to the agreed CISA / SOC collection (not called in tests)."""
    import urllib.request
    req = urllib.request.Request(url, json.dumps(bundle).encode(), method="POST", headers={
        "Content-Type": "application/taxii+json;version=2.1", "Authorization": f"Bearer {token}"})
    return urllib.request.urlopen(req, timeout=10).status


BILLING = queue.Queue()


def _biller():
    while True:
        job, outcome = BILLING.get()
        try: settle(job, outcome)            # $BTC + $BCH, off the hot path
        finally: BILLING.task_done()


threading.Thread(target=_biller, daemon=True).start()


def analyze(dev, iot, cap, now=None, do=lambda a: None, severity="high"):
    """One IoT request on one device: detect -> direction -> distance -> contain -> report."""
    job, _ = admit(iot, "sub-second check", severity, True)
    if job is None: return {"status": "DECLINED"}
    if not dev.registry_ok():
        return {"status": "REFUSED: known-device list tampered"}
    reg = {"devices": {e["name"]: {"mod": e["mod"], "u": e["u"], "freq": e["freq"]}
                       for e in dev.registry}}
    with Device._lock:                       # v2 core, on THIS device's memory + key
        V.EVENTS, V.KEY = dev.events, dev.key
        reg["sig"] = V.sign(reg["devices"])
        settle_now, V.settle = V.settle, lambda j, o: (None, None)
        V.look, V.is_quiet = look3, quiet3
        try: out, _, _ = V.analyze_stream(job, cap, dev.rig, reg, now)
        finally: V.settle, V.look, V.is_quiet = settle_now, LOOK_V2, QUIET_V2
    for m in [m for m in out.get("modalities", []) if not dev.directional[m]]:
        out["modalities"].remove(m)          # rig can't give direction for it
        out.setdefault("intensity_only", []).append(m)
    if out["status"] != "NOT ATTACK" and "sound" in out.get("modalities", []):
        out["distance_m"], out["range"] = distance(clean(cap["sound"]), dev.rig["mics"],
                                                   np.array(out["direction_xyz"]))
    out["contain_now"], out["queued"] = contain(out, do)
    body = {k: v for k, v in out.items() if k not in ("biometrics", "evidence_sig")}
    sha = hashlib.sha256(json.dumps(canonical(body), sort_keys=True).encode()).hexdigest()
    out["evidence_sig"] = dev.sign(body)
    if out["status"] == "ATTACK":
        out["stix"] = stix(dev, out, sha)    # -> send_taxii(...) to CISA / SOC
    BILLING.put((job, "no_attack" if out["status"] == "NOT ATTACK" else "attack"))
    return out
''')


# ==========================================================================
# SECTION 5 · v4 · LIVING OFF THE LAND ON LEGITIMATE SIGNAL TECH · FILTER · LIBRARY
# ==========================================================================
_load("multimodal_v4", r'''
"""BTC Complex · satoshi complex · v4: LIVING OFF THE LAND (LotL).
Also searches LEGITIMATE signal tech (example: a stroke-prevention app driving
a speaker, light or wearable) for a living-off-the-land attack: software that
hijacks the real device to deploy patterns at the public through it.
  none found  -> confirm NONE to the requesting IoT
  detected    -> notify the IoT (details only under its agreement with that
                 medical app) -> report to satoshi complex REAL-TIME COMMAND,
                 which notifies the company, the coder or AI in charge of the
                 legitimate tech, and CISA or another defense office.
  filter      -> where possible, ask the IoT deploying the legitimate tech to
                 FILTER OUT the attack while keeping the legitimate medical tech,
                 check it worked, then the same notifications (+ the IoT maker).
  library     -> shared data + neural-net library of living-off-the-land
                 patterns and filters, for any IoT with that undetected weakness.
Builds on v3 (all of v1-v3 still runs). Data units = $BTC, model runs = $BCH."""
import time
from dataclasses import dataclass, field
import numpy as np
import multimodal_v3 as V3
import multimodal_v2 as V
from multimodal_v2 import unit, angle
from integrated import admit, handle_data, model_run, clean, FS as FS_S


@dataclass
class LegitTech:            # the device's approved, signed profile (from its maker)
    name: str
    company: str
    in_charge: str                  # the coder or AI responsible for it
    mod: str                        # sound / light / kinetic
    u: list                         # where it sits, from the array
    band_hz: tuple                  # approved carrier / flicker band
    pulse_hz: tuple                 # approved on/off rate
    max_level: float                # approved maximum (calibrated at install)
    schedule_h: tuple = (0, 24)     # approved hours of use
    pattern: str = "pulses"         # what it should look like
    firmware_sha: str = ""          # signed firmware the maker shipped
    agreement: set = field(default_factory=set)   # IoT ids with a data agreement
    model: str = ""                 # device model, for the shared library
    iot_maker: str = ""             # maker of the IoT that deploys it
    sig: str = ""


def register(dev, t):
    """Maker-supplied profile, signed on this device (a changed profile is refused)."""
    t.sig = dev.sign({k: v for k, v in t.__dict__.items() if k not in ("sig", "agreement")})
    return t


def profile_ok(dev, t):
    return t.sig == dev.sign({k: v for k, v in t.__dict__.items() if k not in ("sig", "agreement")})


def edge_direction(env, pos, v, fs):
    """Direction from the timing of on/off edges (plain cross-correlation, sub-sample)."""
    rows, d, n = [], [], 2 * len(env)
    E0 = np.fft.rfft(env[:, 0], n)
    for i in range(1, len(pos)):
        cc = np.fft.irfft(np.fft.rfft(env[:, i], n) * np.conj(E0), n)
        m = int(np.ceil(fs * np.linalg.norm(pos[i] - pos[0]) / v)) + 1
        cc = np.concatenate((cc[-m:], cc[:m + 1])); k = int(np.argmax(cc))
        y0, y1, y2 = cc[max(k - 1, 0)], cc[k], cc[min(k + 1, len(cc) - 1)]
        rows.append(pos[0] - pos[i])
        d.append(v * (k - m + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)) / fs)
    u, *_ = np.linalg.lstsq(np.array(rows), np.array(d), rcond=None)
    return unit(u), max(0.0, 1 - abs(np.linalg.norm(u) - 1))


def measure(mod, xw, rig):
    """What the signal really is: frequency, on/off rate, level, shape, direction."""
    if mod == "sound":
        x = clean(xw); fs = FS_S
        keep = V.burst_mask(x[:, 0], fs) > 0
        u, coh = V.direction(x[keep] if keep.sum() > 256 else x, rig["mics"], V.I.C, fs)
        ch = 0; level = float(np.sqrt(np.mean(x ** 2)))
    else:
        base = rig["kin_base"] if mod == "kinetic" else rig["light_base"]
        fs = V.FS_K if mod == "kinetic" else V.FS_L
        x = base.filt(xw); ac = np.sqrt(np.mean(x ** 2, axis=0)); ch = int(np.argmax(ac))
        if mod == "kinetic": u, coh = V.tdoa_direction(x, rig["accel"], V.V_K, fs)
        else:
            v = ac @ V.NORMALS; u = unit(v); coh = float(v @ v / (ac @ ac + 1e-12))
        level = float(ac.max() / (xw.mean() + 1e-9)) if mod == "light" else float(ac.max())
    s = np.abs(np.fft.rfft(x[:, ch]))
    freq = float(np.argmax(s[1:]) + 1) * fs / len(x)
    k = int(np.argmax(s[1:])) + 1
    w = max(3, int(0.01 * k))                                   # within 1% of the tone
    if mod != "light" and np.sum(s[max(1, k - w): k + w + 1] ** 2) > 0.5 * np.sum(s ** 2):
        # a narrow tone repeats every cycle, so its timing is ambiguous: use its on/off edges
        from scipy.signal import butter, sosfilt
        env = sosfilt(butter(2, 2000 if mod == "sound" else 200, fs=fs, output="sos"), np.abs(x), axis=0)
        env = env - env.mean(0)
        pos, v = (rig["mics"], V.I.C) if mod == "sound" else (rig["accel"], V.V_K)
        u, coh = edge_direction(env, pos, v, fs)
    reg, rate = V.periodicity(np.clip(V.envelope(x[:, ch], fs), 0, None), 400)
    eng, why = V.engineered(x[:, ch], fs)
    p = s ** 2 / (np.sum(s ** 2) + 1e-12)
    return {"freq": freq, "rate": rate, "reg": reg, "level": level, "u": u, "coh": coh,
            "shape": why.split()[0] if eng >= 0.5 else "steady", "p": p, "df": fs / len(x)}


def outside(t, m):
    """Share of the signal's energy OUTSIDE its approved band (an attack riding on top)."""
    f = np.arange(len(m["p"])) * m["df"]
    return float(np.sum(m["p"][(f < t.band_hz[0]) | (f > t.band_hz[1])]))


ALIKE = {"pulses": {"pulses", "coded"}, "coded": {"pulses", "coded"}}   # on/off = on/off


def check_tech(t, ms, hour, fw_sha):
    """Compare what the legitimate tech is doing with what it is approved to do."""
    mine = [m for m in ms if m["coh"] >= 0.6 and angle(m["u"], t.u) < 25
            and m["level"] > 0.2 * t.max_level]
    if not mine:
        return [], 0.0
    dev = []
    f = np.median([m["freq"] for m in mine]); r = np.median([m["rate"] for m in mine])
    lv = max(m["level"] for m in mine)
    if not t.band_hz[0] <= f <= t.band_hz[1]:
        dev.append(f"out of its approved band: {f:,.0f} Hz (approved {t.band_hz[0]:,}-{t.band_hz[1]:,})")
    regular = [m for m in mine if m["reg"] > 0.5]
    if regular and not t.pulse_hz[0] <= r <= t.pulse_hz[1]:
        dev.append(f"pulse rate {r:.1f}/s (approved {t.pulse_hz[0]}-{t.pulse_hz[1]})")
    if lv > t.max_level:
        dev.append(f"level {lv / t.max_level:.1f}x its approved maximum")
    out_share = float(np.median([outside(t, m) for m in mine]))
    if out_share > 0.2:
        dev.append(f"{100 * out_share:.0f}% of its energy is outside its approved band")
    shapes = {m["shape"] for m in mine} - {"steady"} - ALIKE.get(t.pattern, {t.pattern})
    if shapes:
        dev.append(f"pattern it should never make: {', '.join(sorted(shapes))}")
    if not t.schedule_h[0] <= hour < t.schedule_h[1]:
        dev.append(f"running outside its approved hours ({hour}:00)")
    if fw_sha and t.firmware_sha and fw_sha != t.firmware_sha:
        dev.append("firmware is not the maker's signed version")
    return dev, len(mine) / len(ms)


def notices(t, iot_id, devs, report_sig, filt=""):
    """Who is told what. Medical details go to the IoT only under its agreement."""
    if not devs:
        return {"iot": f"CONFIRMED NONE: no living-off-the-land attack found in '{t.name}'"}
    iot = (f"PROBLEM with '{t.name}': " + "; ".join(devs) +
           ". Do not rely on it until the maker confirms; ask your clinician before stopping any treatment"
           if iot_id in t.agreement else
           "A safety problem was found with a device near you. Its maker has been told; "
           "ask your clinician before stopping any treatment")
    msg = (f"Detected software-based attack deploying patterns at the public through "
           f"legitimate tech '{t.name}': " + "; ".join(devs) + (f" [{filt}]" if filt else ""))
    return {"iot": iot,
            "command": {"to": "satoshi complex REAL-TIME COMMAND", "msg": msg, "sig": report_sig},
            "company": {"to": t.company, "msg": msg},
            "iot_maker": {"to": t.iot_maker or "maker of the IoT", "msg": msg},
            "in_charge": {"to": t.in_charge, "msg": msg + ". Please confirm and push a signed fix"},
            "defense": {"to": "CISA (or the country's defense office)", "msg": msg, "format": "STIX 2.1"},
            "contain": "maker's safe mode only if the agreement allows; never just cut a device "
                       "someone's treatment depends on"}


SHAPES = ["steady", "pulses", "sweep", "coded", "bursts"]


def features(t, m):
    c, p = sum(t.band_hz) / 2, sum(t.pulse_hz) / 2
    return [m["freq"] / c, m["rate"] / p, m["reg"], m["level"] / t.max_level, outside(t, m)] + \
           [float(m["shape"] == k) for k in SHAPES]


class Library:
    """Shared, anonymous: per device model, what its attacks look like + filters
    that worked + a small neural net. Any IoT with the same model can use it."""
    def __init__(self):
        self.X, self.y, self.filters, self.nets = {}, {}, {}, {}

    def learn(self, t, ms, attack_windows, spec=None, worked=False):
        X = self.X.setdefault(t.model, []); y = self.y.setdefault(t.model, [])
        for i, m in enumerate(ms):
            if m["coh"] >= 0.6 and angle(m["u"], t.u) < 25:
                X.append(features(t, m)); y.append(int(i in attack_windows))
        if spec and worked: self.filters[t.model] = spec
        if len(set(y)) == 2 and len(y) >= 12:
            from sklearn.neural_network import MLPClassifier
            self.nets[t.model] = MLPClassifier((16,), max_iter=2000, random_state=0).fit(X, y)

    def risk(self, t, ms):
        net = self.nets.get(t.model)
        X = [features(t, m) for m in ms if m["coh"] >= 0.6 and angle(m["u"], t.u) < 25]
        return float(np.mean(net.predict_proba(X)[:, 1])) if net is not None and X else None


def window_attacks(t, ms):
    """Which windows broke the approved profile (used to teach the library)."""
    return {i for i, m in enumerate(ms) if m["coh"] >= 0.6 and angle(m["u"], t.u) < 25 and (
        not t.band_hz[0] <= m["freq"] <= t.band_hz[1] or m["level"] > t.max_level
        or outside(t, m) > 0.2 or m["shape"] not in {"steady"} | ALIKE.get(t.pattern, {t.pattern}))}


def filter_spec(t, lib):
    """Keep ONLY the approved therapy: its band, its level; drop everything else."""
    spec = {"keep_band_hz": t.band_hz, "max_level": t.max_level, "mod": t.mod}
    return {**spec, **lib.filters.get(t.model, {})} if lib else spec


def apply_filter(x, spec, fs):
    """What the IoT runs: band-keep filter + level limiter (example)."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = spec["keep_band_hz"]
    y = sosfiltfilt(butter(6, [lo, hi], btype="band", fs=fs, output="sos"), x, axis=0)
    r = np.sqrt(np.mean(y ** 2))
    return y * min(1.0, 0.95 * spec["max_level"] / (r + 1e-12))


def analyze(dev, iot, cap, techs, hour=None, fw=None, now=None, filter_iot=None, lib=None):
    """v3 check + LotL search of every legitimate signal tech in range."""
    hour = time.localtime().tm_hour if hour is None else hour
    fw = fw or {}
    for t in techs:                                   # legit tech isn't an "unknown" source
        if not profile_ok(dev, t):
            return {"status": f"REFUSED: profile of '{t.name}' was changed"}
    known = {e["name"] for e in dev.registry}
    for t in techs:
        if t.name not in known:
            dev.add_benign(t.name, t.mod, t.u, sum(t.band_hz) / 2)
    out = V3.analyze(dev, iot, cap, now)              # everything from v1-v3
    job, _ = admit(iot, "living-off-the-land search", "high", True)
    fs = {"sound": FS_S, "kinetic": V.FS_K, "light": V.FS_L}
    found, lotl = [], {}
    for t in techs:
        ms = []
        for s in np.arange(0, 2.0 - V.WIN + 1e-9, V.HOP):
            xw = cap[t.mod][int(s * fs[t.mod]): int((s + V.WIN) * fs[t.mod])]
            ms.append(model_run(job, measure, t.mod, xw, dev.rig))        # $BCH
        devs, share = check_tech(t, ms, hour, fw.get(t.name, ""))
        p = model_run(job, lib.risk, t, ms) if lib else None             # $BCH
        if p is not None and p > 0.8 and not devs:
            devs.append(f"library: matches a known attack on this model ({100 * p:.0f}%)")
        filt = ""
        if devs and filter_iot:                       # ask the IoT to filter the attack out
            spec = filter_spec(t, lib)
            y = filter_iot(spec)                      # None = it can't / won't filter
            if y is None:
                filt = "IoT can't filter: notified only"
            else:
                ms2 = [measure(t.mod, y[int(s * fs[t.mod]): int((s + V.WIN) * fs[t.mod])], dev.rig)
                       for s in np.arange(0, 2.0 - V.WIN + 1e-9, V.HOP)]
                left, kept = check_tech(t, ms2, 12, "")      # hour/firmware judged above
                ok = kept > 0.5 and not left
                filt = ("FILTERED: attack removed, medical tech maintained" if ok
                        else "filter tried: attack NOT fully removed")
                if lib: lib.learn(t, ms2, set(), spec, ok)
        if lib: lib.learn(t, ms, window_attacks(t, ms) if devs else set())   # $BTC data
        rep = {"tech": t.name, "deviations": devs, "seen_in": f"{100 * share:.0f}% of windows",
               "filter": filt}
        rep_sig = dev.sign(rep)
        lotl[t.name] = {**rep, "notify": notices(t, iot.id, devs, rep_sig, filt)}
        if devs:
            found.append(t.name)
            if out.get("stix"):                       # add LotL detail to the CISA report
                out["stix"]["objects"][1]["x_btcc"]["lotl"] = rep
            else:
                out["stix"] = V3.stix(dev, {**out, "confidence": min(1.0, 0.5 + 0.15 * len(devs)),
                                            "direction_xyz": list(np.round(t.u, 2)),
                                            "fingerprint": rep_sig[:12]}, rep_sig)
                out["stix"]["objects"][1]["x_btcc"]["lotl"] = rep
    handle_data(job, {k: v["deviations"] for k, v in lotl.items()}, True, True)
    V3.BILLING.put((job, "attack" if found else "no_attack"))
    out["lotl"] = lotl
    out["status"] = ("LIVING-OFF-THE-LAND ATTACK" if found else out["status"])
    return out
''')


# ==========================================================================
# SECTION 6 · v5 + v6 · SATOSHI COMPLEX · REVERSE-ENGINEER DETECTION + RESPONSE
# ==========================================================================
import re, time, hmac, hashlib, json
from concurrent.futures import TimeoutError as Late
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import numpy as np
import multimodal_v4 as V4
import multimodal_v3 as V3
from multimodal_v2 import unit, angle

# ---------------------------------------------------------------- 1 · plans
PLANS = {"5":  {"usd": Decimal("5.00"),  "units": 2_000,  "pay_in": ("USD", "BCH")},
         "25": {"usd": Decimal("25.00"), "units": 12_500, "pay_in": ("USD", "BTC", "BCH", "other top blockchain")}}
BUDGET_S = 0.9                                 # the answer always goes out inside 1 second
HOLD_MONTHS = {"paid": 12, "free": 36}       # free: data held longer, until profit share repays


@dataclass
class Account:
    iot_id: str
    units: int = 0
    plan: str = ""
    owed_usd: Decimal = Decimal("0.00")        # exact money: no float rounding
    apr: Decimal = Decimal("0.0")
    profit_share: bool = False
    hold_months: int = HOLD_MONTHS["paid"]
    history: list = field(default_factory=list)

    def buy(self, plan, currency, on_credit=False, offered_aprs=(Decimal("0.0"),)):
        if plan not in PLANS: raise ValueError(f"unknown plan: {plan}")
        p = PLANS[plan]
        if currency not in p["pay_in"]:
            raise ValueError(f"${p['usd']} plan is paid in {', '.join(p['pay_in'])}")
        self.units += p["units"]; self.plan = plan
        if on_credit:                               # pay back later, lowest to no interest
            self.owed_usd += p["usd"]
            self.apr = min(Decimal(str(a)) for a in offered_aprs) if offered_aprs else Decimal("0.0")
        self.history.append((plan, currency, "credit" if on_credit else "paid"))

    def spend(self, n):
        if self.profit_share: return 0              # free: covered by profit share later
        used = min(n, self.units); self.units -= used
        return used


def is_emergency(req):
    """Sub-second triage from what the request already says (no extra data)."""
    flags = [req.get("severity") == "emergency", req.get("person_in_danger", False),
             req.get("active_signal", False), req.get("vital_alarm", False)]
    return any(flags), [k for k, f in zip(("severity", "person in danger", "active signal",
                                           "vital alarm"), flags) if f]


def offer(acct, req):
    em, why = is_emergency(req)
    if em:
        acct.profit_share = True; acct.hold_months = HOLD_MONTHS["free"]
        return {"offer": "FREE", "why": why, "hold_months": acct.hold_months,
                "terms": "free now · monetizable data held longer until profit sharing repays it"}
    if acct.units > 0:
        return {"offer": f"plan ${acct.plan}", "units_left": acct.units}
    return {"offer": "choose", "options": [
        "$5 in USD or $BCH = 2,000 data + neural-network units",
        "$25 in BTC, BCH, USD or another top blockchain = 12,500 units",
        "either one on credit, paid back at the lowest to no interest"]}


# ------------------------------------------------- 2 · reverse engineering
@dataclass
class Query:                # one point of the IoT's own safety / APT query
    ask: str                # what it wants done or known
    data: str               # what data it needs for that
    every_s: float = 1.0    # how often
    send_to: str = "self"   # where the answer goes
    purpose: str = "user safety"
    device_type: str = "generic"   # e.g. camera, smart_lock, smart_bed: what it normally does


OBJECTIVES = [  # what software-based attacks try to do (MITRE ATT&CK-style tactics)
    ("know when the home is empty / when people sleep", "reconnaissance", .90,
     r"\b(occupan\w*|anyone home|home empty|away|presence|asleep|sleep schedule|routine)\b"),
    ("track where a person is", "collection", .80, r"\b(location|gps|which room|where is|position|whereabouts)\b"),
    ("map the building + its devices", "discovery", .80,
     r"\b(floor ?plan|map the|device list|scan (the )?network|open ports|camera list)\b"),
    ("record audio / video", "collection", .85, r"\b(record|microphone|audio|camera|video|listen)\b"),
    ("get credentials, codes or keys", "credential access", .95,
     r"\b(password|passcode|pin|token|api key|unlock code|credential)\b"),
    ("switch off or quiet the defenses", "defense evasion", 1.0,
     r"\b(disable|turn off|silence|mute|bypass|pause) (the )?(alarm|camera|sensor|alert|monitor|update|log)"),
    ("get physical access", "impact", .90, r"\b(unlock|open (the )?(door|garage|gate|window))\b"),
    ("send data out to an unknown place", "exfiltration", .80, r"\b(upload|export|forward|mirror|stream out)\b"),
    ("time an attack by stress or health", "collection", .70,
     r"\b(heart ?rate|cortisol|hrv|stress level|sleep stage|blood pressure|medication)\b"),
    ("stay in the system", "persistence", .90, r"\b(admin|root|install|auto-?start|keep access|new user)\b"),
    ("deploy signals at people", "impact", .90,
     r"\b(emit|play (a )?tone|ultrasonic|flicker|strobe|vibrate (the )?(floor|bed|wall))\b"),
]

EXPECTED = {   # what a device type normally does: cuts false alarms (only if the answer stays home)
    "camera": {"record audio / video"}, "baby_monitor": {"record audio / video"},
    "doorbell": {"record audio / video"}, "smart_lock": {"get physical access"},
    "smart_bed": {"time an attack by stress or health"}, "medical": {"time an attack by stress or health"},
    "thermostat": {"know when the home is empty / when people sleep"},
}

SAFETY_NEEDS = {  # the least each real safety check needs
    "user safety": r"\b(fall|smoke|fire|carbon monoxide|co alarm|flood|gas|signal|anomal\w*|direction|summary|yes/no|alert)\b",
    "fall detection": r"\b(fall|accelerat\w*|impact|summary|yes/no)\b",
    "apt detection": r"\b(signal|pattern|anomal\w*|direction|firmware hash|summary|yes/no)\b",
}
TRUSTED_DEST = {"self", "satoshi complex", "user", "user's phone"}


def score(q):
    """0-100: how much this ONE point is the same as an attack objective."""
    text = f"{q.ask} {q.data}".lower()
    hits = [(o, t, w) for o, t, w, rx in OBJECTIVES if re.search(rx, text, re.I)]
    if not hits:
        return 0, []
    o, t, w = max(hits, key=lambda h: h[2])
    trusted = q.send_to.lower() in TRUSTED_DEST
    normal = o in EXPECTED.get(q.device_type.lower(), set()) and trusted
    s = 0.6 * w * (0.4 if normal else 1.0)
    why = [f"same as attack objective: {o} ({t})" + (" · normal for this device" if normal else "")]
    need = SAFETY_NEEDS.get(q.purpose, SAFETY_NEEDS["user safety"])
    if not re.search(need, q.data.lower()):
        s += 0.25; why.append("more than the safety check needs")
    if not trusted:
        s += 0.20; why.append(f"answer goes to '{q.send_to}'")
    if q.every_s < 60 and t in ("reconnaissance", "collection"):
        s += 0.10; why.append(f"continuous (every {q.every_s:g} s)")
    if "raw" in q.data.lower():
        s += 0.10; why.append("raw, not a summary")
    if len(hits) > 1:
        s += 0.05 * (len(hits) - 1); why.append(f"{len(hits)} attack objectives at once")
    return min(100, round(100 * s)), why


def reverse_engineer(queries, threshold=60):
    pts = []
    for q in queries:
        s, why = score(q)
        pts.append({"point": q.ask, "score": s, "why": why})
    bad = [p for p in pts if p["score"] >= threshold]
    if not bad:
        return {"answer": "NO: that is not the case", "points": pts}
    share = len(bad) / len(pts)
    how_many = "ALL" if share == 1 else "MOST" if share > 0.5 else "SOME"
    return {"answer": f"YES: {how_many} ({len(bad)} of {len(pts)}) of the query points are the same as "
                      f"objectives of a software-based attack", "points": pts}


# ------------------------------------- 3 · find ALL the local relevant IoTs
HIDDEN_KINDS = {  # smart devices that don't look like smart devices
    "smart bed": r"\b(bed|mattress|sleep ?(number|base)|adjustable base)\b",
    "smart table / desk": r"\b(desk|table|standing|sit-?stand)\b",
    "smart chair / sofa": r"\b(chair|recliner|sofa|couch|massage)\b",
    "smart mirror": r"\bmirror\b",
    "smart lamp / bulb": r"\b(bulb|lamp|light ?strip|led)\b",
    "smart plug / outlet": r"\b(plug|outlet|socket|switch)\b",
    "smart appliance": r"\b(fridge|oven|washer|dryer|dishwasher|vacuum|fan|purifier|humidifier)\b",
    "toy / wearable": r"\b(toy|watch|band|ring|tracker|collar)\b",
}
OBVIOUS = r"\b(speaker|camera|thermostat|hub|tv|doorbell|router|phone|tablet|laptop)\b"


def classify(dev):
    text = f"{dev.get('name', '')} {dev.get('maker', '')} {dev.get('services', '')}".lower()
    if re.search(OBVIOUS, text): return "obvious", "smart device"
    for kind, rx in HIDDEN_KINDS.items():
        if re.search(rx, text): return "hidden", kind
    return "hidden", "unknown smart device"     # announces itself, but says nothing clear


def discover(inventory, scan, agreements):
    """Request the IoT's own device list AND scan what is announcing itself
    (Wi-Fi mDNS / SSDP, Bluetooth LE, Zigbee / Thread / Matter), merge them,
    flag the ones that aren't easy to recognize, and where necessary
    establish communication: only through the device's own published channel,
    with the owner's / maker's agreement; otherwise listen-only."""
    seen = {}
    for d in inventory: seen[d["id"]] = {**d, "from": "IoT list"}
    for d in scan:
        seen[d["id"]] = {**seen.get(d["id"], {}), **d,
                         "from": "both" if d["id"] in seen else "scan only (the IoT didn't list it)"}
    out = []
    for d in seen.values():
        how, kind = classify(d)
        if d.get("id") in agreements:
            link = f"connected via {agreements[d['id']]}"
        elif how == "hidden" or d["from"].startswith("scan only"):
            link = "communication requested (owner / maker approval)"
        else:
            link = "known to the IoT"
        out.append({"id": d["id"], "name": d.get("name", "?"), "kind": kind, "recognizable": how == "obvious",
                    "found_by": d["from"], "where": d.get("where"), "link": link})
    return out


# ------------------- 5 · inanimate objects with a part of interest to a signal
SIGNAL_PARTS = {  # part -> the senses an attack could use it for (carry, store, re-radiate)
    "remote-frequency charging": ("rf", "kinetic"), "wireless charging coil": ("rf", "kinetic"),
    "rfid / nfc tag": ("rf",), "antenna / cable run": ("rf",),
    "metal panel / pipe / frame": ("sound", "kinetic", "rf"), "spring / mattress coil": ("kinetic", "rf"),
    "glass / mirror / window film": ("light", "sound"), "led driver / dimmer": ("light", "rf"),
    "solar panel / photocell": ("light",), "piezo / buzzer": ("sound", "kinetic"),
    "motor / fan / pump": ("kinetic", "sound"), "hollow cavity / duct": ("sound",),
}


def signal_objects(objects, senses=None, direction=None, max_deg=30):
    """Objects (from the IoT's data or our scan) with a part an attack could use;
    'in play' = it fits a detected sense AND sits on the signal's line (or it's a field)."""
    out = []
    for o in objects:
        parts = [p for p in o.get("parts", []) if p in SIGNAL_PARTS]
        use = sorted({m for p in parts for m in SIGNAL_PARTS[p]})
        if not use: continue
        out.append({"name": o["name"], "parts": parts, "senses": use, "where": o.get("where"),
                    "in_play": bool(senses) and bool(set(use) & set(senses)) and (
                        direction is None or o.get("where") is None or
                        angle(unit(np.array(o["where"])), unit(np.array(direction))) < max_deg)})
    return out


# ----------------------- 6 · when an attack is detected: field, source, response
def gauge(sig, cap, room):
    """Direction + strength (dB above the room's own level) + linear or field."""
    db = {}
    for m in ("sound", "kinetic", "light"):
        if m in cap and m in room:
            a, b = np.sqrt(np.mean(np.square(cap[m]))), np.sqrt(np.mean(np.square(room[m]))) + 1e-12
            db[m] = round(float(20 * np.log10(a / b + 1e-12)), 1)
    d = sig.get("direction_xyz")
    return {"direction_xyz": d, "strength_db": db, "range": sig.get("range"),
            "shape": "linear (beam from one direction)" if d is not None else "field (no single direction)"}


def origin(direction, rng, devices, objects, transceivers, max_deg=15):
    """What sits on the line the signal came from: nearby IoTs + objects, then
    known far transceivers (e.g. from radio scans / public registries)."""
    if direction is None: return {"origin": "field: no single line", "candidates": []}
    u = unit(np.array(direction)); c = []
    for kind, items in (("IoT", devices), ("object", objects), ("distant transceiver", transceivers)):
        for it in items:
            if it.get("where") is None: continue
            a = angle(unit(np.array(it["where"])), u)
            if a < max_deg: c.append({"name": it["name"], "kind": kind, "off_line_deg": round(a, 1)})
    local = rng is not None and "not" not in str(rng)
    far = lambda x: x["kind"] == "distant transceiver"
    c.sort(key=lambda x: (far(x) if local else not far(x), x["off_line_deg"]))
    best = c[0]["name"] if c else "unknown: nothing known on that line"
    return {"origin": best, "local": local, "candidates": c}


def boosting(levels, now_n=3):
    """Multiples of the level the IoT observes for the subject: now vs its own usual."""
    x = np.asarray(levels, float)
    if len(x) < now_n + 5: return {"boost": 1.0, "boosting": False}
    base = np.median(x[:-now_n]) + 1e-12
    k = float(np.mean(x[-now_n:]) / base)
    steps = int(np.sum(np.diff(x[-(now_n + 1):]) > 0))
    return {"boost": round(k, 1), "boosting": k >= 2, "rising_steps": steps,
            "note": f"x{k:.1f} the usual level" + (" and still rising" if steps == now_n else "")}


INFRA = {"public", "finance", "legal", "gov"}


def severe(req, rev, boost, attack):
    infra = set(req.get("infrastructure", [])) & INFRA
    hi = max([p["score"] for p in rev["points"]] or [0])
    return bool(infra) and (attack or hi >= 60) or boost.get("boost", 1) >= 3 or hi >= 95, sorted(infra)


NOT_ALLOWED = {"raw_audio", "raw_video", "raw_biometrics", "health", "name", "address", "query_text"}


def obfuscate(v, salt=b"btcc-cisa"):
    return "anon-" + hmac.new(salt, str(v).encode(), hashlib.sha256).hexdigest()[:12]


def strip(d):
    if isinstance(d, dict): return {k: strip(v) for k, v in d.items() if k not in NOT_ALLOWED}
    if isinstance(d, list): return [strip(v) for v in d]
    return d


def cisa_packet(iot_id, req, sig, field_, src, rev, boost, sev, filt, stix=None):
    """Required + necessary + obfuscated, minus what the law doesn't allow stored."""
    pkt = {"required": {"status": sig.get("status"), "modalities": sig.get("modalities"),
                        "direction_xyz": field_["direction_xyz"], "shape": field_["shape"],
                        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "severe": sev[0],
                        "infrastructure": sev[1]},
           "necessary": {"strength_db": field_["strength_db"], "origin": src["origin"],
                         "candidates": src["candidates"], "boost": boost, "filter": filt,
                         "iot_query_scores": [{"objective": (p["why"] or ["none"])[0], "score": p["score"]}
                                              for p in rev["points"]]},
           "obfuscated": {"iot": obfuscate(iot_id), "area": obfuscate(req.get("area", "")),
                          "user": obfuscate(req.get("user", ""))},
           "stix": stix}
    return strip({**pkt, **{k: req[k] for k in NOT_ALLOWED if k in req}})   # law-blocked keys removed


def filter_request(src, lib_models, source_can_filter):
    """Ask the source to filter the needle-in-a-haystack / LotL / other attack."""
    name = src["origin"]
    data = name in lib_models
    if src.get("own_queries"):
        ok = bool(source_can_filter.get(name))
        return {"to": name, "possible": ok, "result": ("block these query points + the IoT maker pushes a signed fix"
                if ok else "the IoT itself may be compromised: can't trust it to filter in real time")}
    if source_can_filter.get(name):
        return {"to": name, "possible": True,
                "how": "library filter + neural net for this model" if data else "source's own filter",
                "result": "requested: filter at the source, in real time"}
    return {"to": name, "possible": False, "result": "can't filter at the source in real time",
            "data_available": data}


def strategy(rev, sig, objs, filt, field_):
    s = []
    bad = {w.split(": ")[1].split(" (")[0] for p in rev["points"] if p["score"] >= 60 for w in p["why"][:1]}
    if "switch off or quiet the defenses" in bad: s.append("keep alarms / sensors ON; any off-switch needs a 2nd confirmation")
    if "get physical access" in bad: s.append("no remote unlock without the user or a verified responder")
    if bad & {"know when the home is empty / when people sleep", "track where a person is"}:
        s.append("stop location / occupancy data going to unknown places; send yes/no summaries only")
    if "record audio / video" in bad: s.append("audio stays on the device; only 'distress yes/no' leaves it")
    if sig.get("status") not in ("NOT ATTACK", None):
        s.append("move people out of the line of the signal" if field_["direction_xyz"] else "move people to another room")
        for o in objs:
            if o["in_play"]: s.append(f"pause / shield '{o['name']}' ({', '.join(o['parts'])})")
    if not filt.get("possible"):
        s.append("SEARCH FOR SECONDARY DEPLOYMENTS: other senses (sound / kinetic / light / RF), other "
                 "bands, other devices + objects, other times of day, other network paths")
    return s


# -------------------------------------------------------- 7 · the full run
def relevant_iots(direction, devices, max_deg=30):
    """IoTs that sit in the direction the attack signal came from."""
    if direction is None: return []
    return [d["name"] for d in devices if d.get("where") is not None
            and angle(unit(np.array(d["where"])), unit(np.array(direction))) < max_deg]


_POOL = ThreadPoolExecutor(8)          # shared: no start-up cost per request


def is_attack(sig):
    return "ATTACK" in str(sig.get("status")) and not str(sig.get("status")).startswith("NOT")


def _respond_sig(sig, iot_id, req, rev, cap, room, devices, objs, levels, transceivers, lib_models, sources):
    direction, attack = sig.get("direction_xyz"), is_attack(sig)
    tell = relevant_iots(direction, devices) if attack else []
    return direction, tell, respond(iot_id, req, rev, sig, cap, room, devices, objs, levels,
                                    transceivers, lib_models, sources, attack)


def run(dev, iot, acct, req, queries, cap, techs=(), inventory=(), scan=(), agreements=None, hour=12,
        objects=(), room=None, levels=(), transceivers=(), lib_models=(), source_can_filter=None,
        follow_up=None):
    """Sub-second: triage + plan, then simultaneously: reverse engineering,
    signal sensing, finding all the local IoTs and the signal-relevant objects;
    then, if an attack is detected, the full response in the same pass."""
    t0 = time.perf_counter()
    if not req.get("waiver"):
        return {"status": "DECLINED: waiver not approved"}
    deal = offer(acct, req)
    if deal["offer"] == "choose":
        return {"status": "NEEDS A PLAN", **deal}
    ex = _POOL                                            # all four at the same time
    f_q = ex.submit(reverse_engineer, queries)
    f_s = ex.submit(V4.analyze, dev, iot, cap, list(techs), hour)
    f_d = ex.submit(discover, list(inventory), list(scan), agreements or {})
    f_o = ex.submit(signal_objects, list(objects))
    rev, devices, objs = f_q.result(), f_d.result(), f_o.result()          # fast: milliseconds
    args = (iot.id, req, rev, cap, room or {}, devices, objs, levels, transceivers,
            set(lib_models), source_can_filter or {})
    try:                                                  # signals: whatever is left of the 0.9 s
        sig = f_s.result(timeout=max(0.05, BUDGET_S - (time.perf_counter() - t0)))
    except Late:                                          # never hold the answer: say so, and
        sig = {"status": "STILL SENSING: full signal result follows", "direction_xyz": None,
               "modalities": [], "lotl": {}}             # send the full result the moment it's in
        if follow_up:
            f_s.add_done_callback(lambda f: follow_up(_respond_sig(f.result(), *args)))
    units = len(queries) + len(devices) + len(objs) + sum(1 for _ in cap) + 9 * 3 + 2  # data + NN units
    used = acct.spend(units)
    direction, tell, resp = _respond_sig(sig, *args)
    return {"seconds": round(time.perf_counter() - t0, 2), "offer": deal,
            "reverse_engineering": rev,
            "signals": {"status": sig.get("status"), "direction_xyz": direction,
                        "range": sig.get("range"), "modalities": sig.get("modalities"),
                        "inform_iots": tell, "lotl": {k: v["deviations"] for k, v in sig.get("lotl", {}).items()}},
            "local_iots": devices, "signal_objects": resp.pop("objects", objs), "response": resp,
            "units_used": units, "units_charged": used, "units_left": acct.units,
            "hold_months": acct.hold_months}


def respond(iot_id, req, rev, sig, cap, room, devices, objs, levels, transceivers, lib_models,
            source_can_filter, attack):
    """Runs only when an attack is detected on the subject of the IoT's request:
    signal patterns at the public, OR the IoT's own data scoring = attack objectives."""
    yes = rev["answer"].startswith("YES")
    boost = boosting(levels)
    if not (attack or yes):
        return {"detected": False, "to_iot": "NO ATTACK: status + results sent; nothing to filter",
                "boost": boost}
    senses = [m for m in sig.get("modalities") or []]
    objs = signal_objects([{"name": o["name"], "parts": o["parts"], "where": o["where"]} for o in objs],
                          senses, sig.get("direction_xyz"))
    field_ = gauge(sig, cap, room) if attack else {"direction_xyz": None, "strength_db": {}, "range": None,
                                                   "shape": "no signal: the IoT's own data"}
    src = (origin(field_["direction_xyz"], field_["range"], devices, objs, transceivers) if attack
           else {"origin": iot_id, "own_queries": True, "candidates": []})
    notify = [c["name"] for c in src["candidates"] if c["kind"] in ("IoT", "distant transceiver")]
    filt = filter_request(src, lib_models, source_can_filter)
    sev = severe(req, rev, boost, attack)
    plan = strategy(rev, sig, objs, filt, field_)
    stix = sig.get("stix")
    pkt = cisa_packet(iot_id, req, sig, field_, src, rev, boost, sev, filt, stix)
    to_iot = {"status": sig.get("status") if attack else "IoT data = attack objectives",
              "reverse_engineering": rev["answer"], "field": field_, "origin": src["origin"],
              "boost": boost, "severe": sev, "filter": filt,
              "filter_possible": "YES" if filt["possible"] else "NO: not in real time at the source",
              "strategy": plan, "sent_to_cisa": sorted(k for k in pkt if pkt[k])}
    return {"detected": True, "objects": objs, "field": field_, "origin": src, "notify_iots": notify,
            "filter": filt, "boost": boost, "severe": sev, "strategy": plan, "cisa": pkt, "to_iot": to_iot}
