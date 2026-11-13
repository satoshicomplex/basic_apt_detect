"""
BTC Complex · satoshi complex · COMMUNICATIONS FRAMEWORK  (v2)
===================================================================

WHAT THIS DOES
    When an IoT device asks satoshi complex to confirm user safety (the
    "requesting IoT"), we may need help from OTHER IoT devices nearby: a
    speaker that can listen, a smart bed that can feel vibration, a bulb that
    can dim. This module finds those helpers, asks them all at once, and never
    waits past a hard deadline.

THE FLOW (one call: ask_all)
    1. relevant()   which peers can provide what this request needs
                    (filtered by zone, matched by sense / model / function)
    2. readiness()  can each one work with us in time?  decided from MEASURED
                    replies, not guesses:
                        SUB_SECOND     p90 reply <= 300 ms -> real time
                        SLOWER         p90 <= 2 s          -> confirmations only
                        UNTESTED       never timed         -> asked, and timed now
                        NO_CHANNEL     no way to reach it  (connection requested)
                        NO_AGREEMENT   owner / maker hasn't opted in
                        COOLING_DOWN   failed repeatedly; skipped for a while
                        TOO_SLOW       too slow for this mode
    3. build_request()  one structured, signed Request per peer; its .text says:
                        - EMERGENCY (a possible real-time incident) or
                          NON-EMERGENCY (a safety confirmation)
                        - that it concerns the requesting IoT's [____] request
                        - what we ask for, and by when
                        - that the peer EARNS from what it shares
    4. send in parallel, collect only what arrives before the deadline
    5. Ledger: credit each helper per use, auditable entry by entry

PAYMENT RULE  (company-wide)
    data shared by a peer (it SENSES)            -> data units,           paid in $BTC
    neural networks / models a peer runs (MODEL) -> neural-network units, paid in $BCH
    functions a peer performs (DO, e.g. mute)    -> billed as neural-network
                                                    units in $BCH (change in PAY)
    Only valid replies that arrive in time are paid.

DEADLINES
    emergency 300 ms · non-emergency 2,000 ms.  ask_all() returns by then,
    whatever happens; late or failed peers are recorded and slowly demoted.

PRIVACY
    The requester is identified only by a pseudonym. Messages carry nothing
    about people beyond what the request needs. Requests are HMAC-signed so a
    peer can verify they came from satoshi complex.

PLUGGING IN
    transport:  any function  send(peer, request) -> dict | None
                (Matter, BLE, a maker's cloud API ...). A reply must be a dict
                with "ok": True to count. Exceptions are caught.
    peers:      from satoshi_complex.discover(); add .can / .zone / .channel
    run it:     python3 satoshi_comms.py   (self-test with simulated peers)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Optional

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
DEADLINE_MS = {"emergency": 300, "non-emergency": 2000}
MIN_SAMPLES_FOR_P90 = 1          # one timed reply is enough to classify
FAILS_BEFORE_COOLDOWN = 3        # consecutive misses before a peer rests
COOLDOWN_S = 60.0


class Tier(str, Enum):
    """What a peer provides, and how it is paid."""
    DATA = "data"                # it senses -> data units ($BTC)
    MODEL = "neural_net"         # it runs a neural net / model -> NN units ($BCH)
    FUNCTION = "function"        # it does something (mute, stop motor)


PAY = {                          # tier -> (currency, unit name, units per use)
    Tier.DATA: ("$BTC", "data unit", 1),
    Tier.MODEL: ("$BCH", "neural-network unit", 1),
    Tier.FUNCTION: ("$BCH", "neural-network unit", 1),
}


class Ready(str, Enum):
    SUB_SECOND = "SUB-SECOND: works with us in real time"
    SLOWER = "SLOWER: fine for non-emergency confirmations"
    UNTESTED = "UNTESTED: will be timed on this request"
    NO_CHANNEL = "CAN'T: no way to reach it (connection requested)"
    NO_AGREEMENT = "CAN'T YET: needs the owner's / maker's agreement"
    COOLING_DOWN = "RESTING: failed repeatedly, retried after cooldown"
    TOO_SLOW_EMERGENCY = "TOO SLOW for this emergency: confirmations only"
    TOO_SLOW = "CAN'T: too slow"


SENDABLE = {Ready.SUB_SECOND, Ready.SLOWER, Ready.UNTESTED}


def _key() -> bytes:
    k = os.environ.get("BTCC_DEVICE_KEY")       # production: TPM / secure element
    if not k:
        raise RuntimeError("no device key: provision it in the TPM / secure element")
    return k.encode()


# --------------------------------------------------------------------------
# peers
# --------------------------------------------------------------------------
@dataclass
class Peer:
    """A nearby IoT that may help. `can` lists what it offers:
    {"sense": ["sound", ...], "model": ["fall detector", ...], "do": ["mute", ...]}"""
    name: str
    kind: str
    can: dict
    zone: str = "default"                    # room / area; "global" = everywhere
    channel: str = ""                        # Matter / BLE / maker API; "" = unreachable
    agreement: bool = False                  # owner / maker opted in
    latency_ms: deque = field(default_factory=lambda: deque(maxlen=50))
    fails_in_row: int = 0
    resting_until: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def p90(self) -> Optional[float]:
        with self._lock:
            if len(self.latency_ms) < MIN_SAMPLES_FOR_P90:
                return None
            x = sorted(self.latency_ms)
        return x[min(len(x) - 1, round(0.9 * (len(x) - 1)))]

    def record(self, ms: float, ok: bool) -> None:
        """Every outcome updates the peer: speed, failures, cooldown."""
        with self._lock:
            self.latency_ms.append(ms)
            self.fails_in_row = 0 if ok else self.fails_in_row + 1
            if self.fails_in_row >= FAILS_BEFORE_COOLDOWN:
                self.resting_until = time.time() + COOLDOWN_S


def relevant(peers, need, zone=None):
    """Peers that can provide something in `need`, with how each item is paid.
    Returns [(peer, [(item, Tier), ...]), ...]."""
    out = []
    for p in peers:
        if zone and p.zone not in (zone, "global"):
            continue
        matched = []
        for item in need:
            if item in p.can.get("sense", []):
                matched.append((item, Tier.DATA))
            elif item in p.can.get("model", []):
                matched.append((item, Tier.MODEL))
            elif item in p.can.get("do", []):
                matched.append((item, Tier.FUNCTION))
        if matched:
            out.append((p, matched))
    return out


def readiness(p: Peer, emergency: bool) -> Ready:
    """Can this peer work with us in time? Order matters: reachability and
    consent first, then rest periods, then measured speed."""
    if not p.channel:
        return Ready.NO_CHANNEL
    if not p.agreement:
        return Ready.NO_AGREEMENT
    if time.time() < p.resting_until:
        return Ready.COOLING_DOWN
    t = p.p90()
    if t is None:
        return Ready.UNTESTED
    if t <= DEADLINE_MS["emergency"]:
        return Ready.SUB_SECOND
    if t <= DEADLINE_MS["non-emergency"]:
        return Ready.TOO_SLOW_EMERGENCY if emergency else Ready.SLOWER
    return Ready.TOO_SLOW


# --------------------------------------------------------------------------
# the request each peer receives
# --------------------------------------------------------------------------
@dataclass
class Request:
    request_id: str
    to: str
    mode: str                    # EMERGENCY / NON-EMERGENCY
    requester: str               # pseudonym of the requesting IoT
    request_type: str            # the "____" in "regarding its ____ request"
    asks: list                   # [(item, tier), ...]
    deadline_ms: int
    offer: dict                  # what the peer earns, per currency
    sent_at: float
    signature: str = ""

    @property
    def text(self) -> str:
        head = ("EMERGENCY · a possible REAL-TIME INCIDENT" if self.mode == "EMERGENCY"
                else "NON-EMERGENCY · a safety CONFIRMATION, nothing is wrong yet")
        pays = " + ".join(f"{n} {unit}{'s' * (n > 1)} in {cur}" for cur, (unit, n) in self.offer.items())
        return (f"satoshi complex -> {self.to}\n{head}\n"
                f"This is regarding the requesting IoT ({self.requester}) and its "
                f"'{self.request_type}' request.\n"
                f"We ask you for: {', '.join(i for i, _ in self.asks)} · "
                f"please answer within {self.deadline_ms} ms.\n"
                f"You EARN from the data, neural networks or functions you share "
                f"({pays}), paid per use for answers in time. Nothing is shared about "
                f"people beyond what the request needs.")


def build_request(peer, matched, requester, request_type, emergency, deadline_ms) -> Request:
    offer = {}
    for _, tier in matched:
        cur, unit, n = PAY[tier]
        u, m = offer.get(cur, (unit, 0))
        offer[cur] = (u, m + n)
    r = Request(request_id=uuid.uuid4().hex[:12], to=peer.name,
                mode="EMERGENCY" if emergency else "NON-EMERGENCY", requester=requester,
                request_type=request_type, asks=[(i, t.value) for i, t in matched],
                deadline_ms=deadline_ms, offer=offer, sent_at=time.time())
    body = json.dumps({**asdict(r), "signature": ""}, sort_keys=True, default=str).encode()
    r.signature = hmac.new(_key(), body, hashlib.sha256).hexdigest()
    return r


def verify_request(r: Request) -> bool:
    """What a peer's SDK calls to check a request really came from us."""
    body = json.dumps({**asdict(r), "signature": ""}, sort_keys=True, default=str).encode()
    return hmac.compare_digest(r.signature, hmac.new(_key(), body, hashlib.sha256).hexdigest())


# --------------------------------------------------------------------------
# who earned what
# --------------------------------------------------------------------------
class Ledger:
    """Append-only record of every payment owed to a helper peer."""

    def __init__(self):
        self.entries, self._lock = [], threading.Lock()

    def credit(self, peer: str, request_id: str, item: str, tier: Tier) -> None:
        cur, unit, n = PAY[tier]
        with self._lock:
            self.entries.append({"peer": peer, "request_id": request_id, "item": item,
                                 "tier": tier.value, "currency": cur, "unit": unit,
                                 "units": n, "time": time.time()})

    def totals(self) -> dict:
        out = {}
        for e in self.entries:
            out.setdefault(e["peer"], {"$BTC": 0, "$BCH": 0})[e["currency"]] += e["units"]
        return out


LEDGER = Ledger()


# --------------------------------------------------------------------------
# the one call
# --------------------------------------------------------------------------
Transport = Callable[[Peer, Request], Optional[dict]]


def _timed(send: Transport, p: Peer, req: Request):
    t = time.perf_counter()
    try:
        reply = send(p, req)
    except Exception as e:                                   # a peer can never crash us
        reply = {"ok": False, "error": type(e).__name__}
    return reply, 1000 * (time.perf_counter() - t)


def ask_all(peers, need, requester, request_type, emergency, send: Transport,
            zone=None, pool: ThreadPoolExecutor | None = None, ledger: Ledger = LEDGER) -> dict:
    """Ask every relevant peer at once. Returns within the deadline, always."""
    mode = "emergency" if emergency else "non-emergency"
    dl = DEADLINE_MS[mode]
    t0 = time.perf_counter()
    rows, jobs = [], {}
    own = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=16)
    try:
        for p, matched in relevant(peers, need, zone):
            ready = readiness(p, emergency)
            row = {"peer": p.name, "kind": p.kind, "zone": p.zone,
                   "asks": [i for i, _ in matched], "readiness": ready.value}
            if ready in SENDABLE:
                req = build_request(p, matched, requester, request_type, emergency, dl)
                row["request"] = req
                jobs[pool.submit(_timed, send, p, req)] = (p, matched, row, req)
            rows.append(row)

        left = dl / 1000 - (time.perf_counter() - t0)            # deadline counts from the start
        done, _ = wait(jobs, timeout=max(0.0, left))

        for f, (p, matched, row, req) in jobs.items():
            if f not in done:
                f.cancel()
                p.record(dl * 2, ok=False)                        # demote slow peers
                row["status"] = "NO ANSWER in time: carried on without it"
                continue
            reply, ms = f.result()
            row["ms"] = round(ms)
            if isinstance(reply, dict) and reply.get("ok"):
                p.record(ms, ok=True)
                row.update(status="ANSWERED in time", reply=reply)
                for item, tier in matched:
                    ledger.credit(p.name, req.request_id, item, tier)
            else:
                p.record(ms, ok=False)
                row["status"] = f"FAILED: {(reply or {}).get('error', 'empty reply')}"
    finally:
        if own:
            pool.shutdown(wait=False, cancel_futures=True)

    answered = sum(r.get("status") == "ANSWERED in time" for r in rows)
    return {"mode": mode.upper(), "deadline_ms": dl,
            "total_ms": round(1000 * (time.perf_counter() - t0)),
            "asked": len(jobs), "answered": answered, "peers": rows}


# --------------------------------------------------------------------------
# self-test:  python3 satoshi_comms.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("BTCC_DEVICE_KEY", "test-only-key")
    peers = [
        Peer("smart speaker", "speaker", {"sense": ["sound"], "do": ["mute"]}, "living", "Matter", True),
        Peer("smart bed", "bed", {"sense": ["kinetic"], "model": ["sleep-stress net"]}, "bedroom", "maker API", True),
        Peer("camera", "camera", {"sense": ["light", "sound"]}, "living", "", False),
        Peer("old hub", "hub", {"sense": ["sound"]}, "global", "cloud API", True, deque([1200, 1100])),
    ]
    delay = {"smart speaker": .04, "smart bed": .12, "old hub": 1.1}
    send = lambda p, r: (time.sleep(delay[p.name]), {"ok": True})[1]
    for em in (True, False):
        out = ask_all(peers, ["sound", "kinetic", "sleep-stress net", "mute"], "anon-hub7",
                      "confirm user safety", em, send)
        print(out["mode"], out["total_ms"], "ms")
        for r in out["peers"]:
            print(f"  {r['peer']:14s} {r['readiness']:50s} {r.get('status', '-')}")
    print("earned:", LEDGER.totals())
