"""
BCH Complex · MEDICAL ERROR PREVENTION
Hardened Reference Framework (v2)
================================================================================

PURPOSE
    Sub-second detection of preventable medical harm from a short observation
    window (0.25–1.0 s). After waiver acceptance and optional payment selection,
    a single call:

    1. Analyzes the window against externally supplied thresholds
       (drug label, venue policy, device profile). Does not invent clinical limits.
    2. Contacts nearby medical IoTs that can act.
    3. Looks for:
         • clearly incorrect dosage that would cause injury
         • pathogen load requiring sanitization
         • air-quality breaches (documented hospital hazard)
         • employee error, including intentional markers
    4. Returns concrete actions the requesting IoT can take + list of peers asked.
    5. Escalates device issues to makers, human issues to the medical business,
       and intentional cases to the Command Center + authorities (lawful data only).
    6. Stores only explicitly allow-listed, non-protected, monetizable facts.

PRIVACY (non-negotiable)
    Protected fields (identity, MRN, diagnosis, raw clinical data, biometric
    hashes, etc.) exist only in memory for the duration of the call. They are
    overwritten and deleted before return. Nothing protected is written to
    storage, logs, or any durable form. A hash of protected data is still
    protected data and is never retained. Configure the protected set and
    retention rules per jurisdiction with counsel.

PAYMENT
    • Data units  → $BTC
    • Neural-net units → $BCH
    Verified emergency = free. Failed or insufficient request = zero charge.

STATUS
    Reference implementation, simulation-tested only. Real deployment requires
    validated models, contractual agreement with each maker and venue, lawful
    authority, and review by counsel.

DISCLAIMER
    Informational only. Not medical advice. Consult a qualified professional.
    AI-generated code may contain errors.

PROVENANCE
    Draft (Claude) -> review (Google) -> hardening (Grok) -> merged. From the
    reviews: the emergency-attestation check (free only with a signed token or a
    confirmed life-threatening finding), the always-run protected-data purge in a
    finally block (deep-copied, nulled and deleted - never mutating the caller),
    per-detector input validation with a fail-safe status, event ids and
    timestamps, and the conservative (weakest-link) certainty for a clean window.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("bch_complex.medical")

MIN_WINDOW_S = 0.25
MAX_WINDOW_S = 1.0

# ---------------------------------------------------------------------------
# 1 · Admission & settlement
# ---------------------------------------------------------------------------
PLANS: Dict[str, Tuple[int, Tuple[str, ...]]] = {
    "5": (2_000, ("USD", "BCH")),
    "25": (12_500, ("USD", "BTC", "BCH", "top chain")),
}


@dataclass
class Account:
    iot_id: str
    units: int = 0
    plan: str = ""
    charged_total: int = 0

    def buy(self, plan: str, currency: str) -> None:
        if plan not in PLANS:
            raise ValueError(f"unknown plan '{plan}'")
        units, accepted = PLANS[plan]
        if currency not in accepted:
            raise ValueError(f"${plan} plan accepts only {', '.join(accepted)}")
        self.units += units
        self.plan = plan

    def settle(
        self,
        data_units_btc: int,
        nn_units_bch: int,
        succeeded: bool,
        verified_emergency: bool,
    ) -> int:
        if verified_emergency or not succeeded:
            return 0
        cost = max(0, data_units_btc) + max(0, nn_units_bch)
        used = min(cost, self.units)
        self.units -= used
        self.charged_total += used
        return used


def verify_emergency(req: Dict[str, Any], findings: Optional[List["Finding"]] = None) -> bool:
    """Emergency is free only with signed attestation or a confirmed life-threatening finding."""
    if not req.get("emergency"):
        return False
    has_attestation = bool(req.get("emergency_attestation_token"))
    has_critical = bool(
        findings and any(f.detected and f.severity == "life-threatening" for f in findings)
    )
    return has_attestation or has_critical


def admit(acct: Account, req: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    if not req.get("waiver"):
        return None, "DECLINED: waiver not approved", False

    if req.get("emergency") and req.get("emergency_attestation_token"):
        return {
            "offer": "PROVISIONAL_EMERGENCY_FREE",
            "terms": "verified emergency: no charge; standard rates if unverified",
        }, "OK", True

    if acct.units > 0:
        return {"offer": f"plan ${acct.plan}", "units_left": acct.units}, "OK", True

    return {
        "offer": "choose_plan",
        "options": [
            "$5 USD/BCH = 2,000 units",
            "$25 USD/BTC/BCH/top chain = 12,500 units",
        ],
    }, "NEEDS_PLAN", False


# ---------------------------------------------------------------------------
# 2 · Privacy – live-only use, explicit purge
# ---------------------------------------------------------------------------
PROTECTED: Set[str] = {
    "name", "mrn", "dob", "ssn", "address", "face", "diagnosis",
    "raw_vitals", "genetic", "clinical_note", "photo", "employee_name",
    "patient_id", "clinician_id", "biometric_hash", "phi", "pii",
}


def for_storage(record: Dict[str, Any], allow_keys: List[str]) -> Dict[str, Any]:
    """Keep only explicitly allow-listed, non-protected facts."""
    allow = {k.lower() for k in allow_keys}
    return {
        k: v for k, v in record.items()
        if k.lower() in allow and k.lower() not in PROTECTED
    }


def extract_and_purge_protected(payload: Any, found: Optional[Set[str]] = None) -> List[str]:
    """Recursively null and delete every protected key. Returns the keys seen."""
    if found is None:
        found = set()
    if isinstance(payload, dict):
        for k in list(payload.keys()):
            if k.lower() in PROTECTED:
                found.add(k)
                payload[k] = None
                del payload[k]
            else:
                extract_and_purge_protected(payload[k], found)
    elif isinstance(payload, list):
        for item in payload:
            extract_and_purge_protected(item, found)
    return sorted(found)


# ---------------------------------------------------------------------------
# 3 · Detectors (thresholds are inputs; fail-safe on bad data)
# ---------------------------------------------------------------------------
class ErrorType(str, Enum):
    DOSAGE = "incorrect dosage"
    PATHOGEN = "pathogen / contamination"
    AIR = "air quality"
    HUMAN = "medical-employee error"


@dataclass
class Finding:
    kind: ErrorType
    detected: bool
    severity: str                       # none | injury | life-threatening
    certainty: float                    # 0.0 – 1.0
    intentional: bool = False
    cause: str = ""                     # device | human
    source: str = ""                    # model or role, never a personal name
    iot_can_do: List[str] = field(default_factory=list)
    detail: str = ""
    valid_check: bool = True


def _certainty(window_s: float, sensor_conf: float, corroborating: int) -> float:
    w = max(0.0, min(1.0, (window_s - MIN_WINDOW_S) / (MAX_WINDOW_S - MIN_WINDOW_S) + 0.4))
    c = max(0.0, min(1.0, sensor_conf))
    cor = min(2, max(0, corroborating))
    return round(min(1.0, 0.5 * w + 0.4 * c + 0.1 * cor), 2)


def check_dosage(order: Dict[str, Any], window_s: float) -> Finding:
    lo, hi = order.get("safe_low"), order.get("safe_high")
    given = order.get("administered")
    if (
        given is None or lo is None or hi is None
        or not isinstance(given, (int, float)) or not isinstance(lo, (int, float))
        or not isinstance(hi, (int, float))
        or lo < 0 or hi < lo or given < 0
    ):
        return Finding(ErrorType.DOSAGE, False, "none", 0.0,
                       detail="invalid or missing dose/range", valid_check=False)

    over = given > hi
    under = given < lo
    if not (over or under):
        return Finding(
            ErrorType.DOSAGE, False, "none",
            _certainty(window_s, order.get("sensor_conf", 0.9), order.get("sources", 1)),
            detail="within supplied label range",
        )

    over_factor = (given / hi) if (over and hi > 0) else 0.0
    under_factor = (lo / given) if (under and given > 0) else (float("inf") if under else 0.0)
    severe = (
        (over and over_factor >= 2.0)
        or (under and under_factor >= 5.0 and order.get("critical_life_support", False))
    )

    return Finding(
        ErrorType.DOSAGE,
        True,
        "life-threatening" if severe else "injury",
        _certainty(window_s, order.get("sensor_conf", 0.9), order.get("sources", 1)),
        cause="device" if order.get("by_pump") else "human",
        source=order.get("pump_model") or order.get("role", "clinician"),
        iot_can_do=(
            ["pause the infusion pump", "alarm + require a second check"]
            if order.get("by_pump")
            else ["alarm the clinician", "require a second nurse to confirm"]
        ),
        detail=(
            f"{order.get('drug', 'drug')} {given}{order.get('unit', '')} "
            f"{'above' if over else 'below'} label range {lo}-{hi}"
        ),
    )


def check_pathogen(samples: Dict[str, Any], window_s: float) -> Finding:
    load = samples.get("pathogen_index")
    thresh = samples.get("sanitize_threshold")
    if (
        load is None or thresh is None
        or not isinstance(load, (int, float)) or not isinstance(thresh, (int, float))
        or thresh <= 0 or load < 0
    ):
        return Finding(ErrorType.PATHOGEN, False, "none", 0.0,
                       detail="invalid or missing pathogen metrics", valid_check=False)

    if load < thresh:
        return Finding(
            ErrorType.PATHOGEN, False, "none",
            _certainty(window_s, samples.get("sensor_conf", 0.85), samples.get("sources", 1)),
        )

    return Finding(
        ErrorType.PATHOGEN,
        True,
        "life-threatening" if load >= 2.0 * thresh else "injury",
        _certainty(window_s, samples.get("sensor_conf", 0.85), samples.get("sources", 1)),
        cause="device" if samples.get("on_equipment") else "human",
        source=samples.get("surface", "surface"),
        iot_can_do=[
            "flag for immediate sanitization",
            "take the item / room out of use",
            "alert environmental services",
        ],
        detail=f"pathogen index {load} ≥ sanitize threshold {thresh} on {samples.get('surface', 'surface')}",
    )


def check_air(readings: Dict[str, Any], window_s: float) -> Finding:
    metrics = ("pm2_5", "co2_ppm", "airborne_pathogen")
    evaluated = [k for k in metrics if k in readings and f"{k}_limit" in readings]
    if not evaluated:
        return Finding(ErrorType.AIR, False, "none", 0.0,
                       detail="no valid air metric/limit pairs", valid_check=False)

    breaches = [k for k in evaluated if readings[k] > readings[f"{k}_limit"]]
    if not breaches:
        return Finding(
            ErrorType.AIR, False, "none",
            _certainty(window_s, readings.get("sensor_conf", 0.8), len(evaluated)),
        )

    severe = any(
        readings[k] >= 2.5 * readings[f"{k}_limit"]
        for k in breaches if readings.get(f"{k}_limit", 0) > 0
    )

    return Finding(
        ErrorType.AIR,
        True,
        "life-threatening" if severe else "injury",
        _certainty(window_s, readings.get("sensor_conf", 0.8), len(breaches)),
        cause="device" if readings.get("hvac_fault") else "human",
        source=readings.get("zone", "ward air"),
        iot_can_do=[
            "boost filtration / fresh-air exchange",
            "alert facilities + the HVAC maker",
            "move vulnerable patients from the zone",
        ],
        detail=f"over limit: {', '.join(breaches)} in {readings.get('zone', 'the ward')}",
    )


LEGIT_CARE = {
    "ordered_high_dose_under_protocol",
    "therapeutic_hypothermia",
    "permissive_hypotension",
    "deliberate_sedation",
    "palliative_protocol",
}


def check_human(action: Dict[str, Any], window_s: float) -> Finding:
    if action.get("protocol_tag") in LEGIT_CARE:
        return Finding(
            ErrorType.HUMAN, False, "none",
            _certainty(window_s, action.get("sensor_conf", 0.8), action.get("sources", 1)),
            detail="recognized protocolized care; not an error",
        )

    deviates = bool(action.get("deviates_from_protocol", False))
    would_injure = bool(action.get("would_cause_injury", False))
    if not (deviates and would_injure):
        return Finding(
            ErrorType.HUMAN, False, "none",
            _certainty(window_s, action.get("sensor_conf", 0.8), action.get("sources", 1)),
        )

    intent_flags = [
        f for f in (
            "unauthorized_override",
            "tampering",
            "credential_mismatch",
            "repeated_after_warning",
            "disabled_safety",
        )
        if action.get(f)
    ]
    intentional = len(intent_flags) >= 1

    return Finding(
        ErrorType.HUMAN,
        True,
        "life-threatening" if action.get("life_threatening") else "injury",
        _certainty(window_s, action.get("sensor_conf", 0.8), action.get("sources", 1)),
        intentional=intentional,
        cause="human",
        source=action.get("role", "staff"),
        iot_can_do=[
            "alarm + require an independent second check",
            "lock the unsafe action pending review",
        ],
        detail=(
            "protocol deviation that would cause injury"
            + (f"; intentional markers: {', '.join(intent_flags)}" if intentional else "")
        ),
    )


# ---------------------------------------------------------------------------
# 4 · Nearby IoTs + upward reporting
# ---------------------------------------------------------------------------
def ask_nearby(nearby_iots: List[Dict[str, Any]], finding: Finding) -> List[Dict[str, Any]]:
    target = finding.kind.name.lower()
    asks = []
    for io in nearby_iots:
        caps = [str(c).lower() for c in io.get("can_help_with", [])]
        if target in caps or "any" in caps:
            asks.append({
                "iot": io.get("name", "unknown_iot"),
                "request": "help prevent this error",
                "provided": {"data": "$BTC data units", "neural_net": "$BCH NN units"},
                "earns": True,
            })
    return asks


def escalate(finding: Finding, venue: Dict[str, Any], prevention: str, event_id: str) -> Optional[Dict[str, Any]]:
    if not finding.detected:
        return None
    if finding.cause == "device":
        to = venue.get("device_makers", {}).get(finding.source, "the device maker")
        level = "device maker"
    else:
        to = venue.get("name", "the medical business")
        level = "medical business"
    return {
        "event_id": event_id,
        "to": to,
        "level": level,
        "about": finding.kind.value,
        "severity": finding.severity,
        "source": finding.source,
        "prevention": prevention,
        "contains_protected_data": False,
    }


def notify_authorities(
    finding: Finding, venue: Dict[str, Any], storable: Dict[str, Any], event_id: str
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "command_center": "BCH Complex Real-Time Command",
        "to": [venue.get("name", "the medical venue"), "the proper authorities"],
        "about": f"suspected intentional {finding.kind.value}",
        "severity": finding.severity,
        "source_role": finding.source,
        "lawful_payload": storable,
        "protected_data_withheld": True,
    }


def _prevention(f: Finding) -> str:
    return {
        ErrorType.DOSAGE: "hard dose limits on the pump + mandatory independent second check",
        ErrorType.PATHOGEN: "tighter sanitization cadence + contamination sensing at point of use",
        ErrorType.AIR: "continuous air monitoring with auto filtration boost + HVAC servicing",
        ErrorType.HUMAN: "role-based locks, independent double-checks, and audit on overrides",
    }[f.kind]


# ---------------------------------------------------------------------------
# 5 · Main orchestrator
# ---------------------------------------------------------------------------
def respond(
    acct: Account,
    req: Dict[str, Any],
    data: Dict[str, Any],
    nearby_iots: Optional[List[Dict[str, Any]]] = None,
    venue: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    def ms() -> float:
        return round((time.perf_counter() - t0) * 1000, 2)

    event_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    nearby_iots = nearby_iots or []
    venue = venue or {}

    working_data = copy.deepcopy(data)
    receipt: Optional[Dict[str, Any]] = None
    storable: Dict[str, Any] = {}
    earners: List[str] = []
    billed = 0
    out: Dict[str, Any] = {"event_id": event_id, "timestamp_utc": timestamp_utc}

    try:
        receipt, status, authorized = admit(acct, req)
        if not authorized:
            out.update({"status": status, "receipt": receipt, "latency_ms": ms()})
            return out

        try:
            window_s = float(req.get("window_s", 0.5))
        except (TypeError, ValueError):
            window_s = 0.5

        if window_s < MIN_WINDOW_S:
            out.update({
                "status": "NEED_MORE_DATA",
                "need_at_least_s": MIN_WINDOW_S,
                "latency_ms": ms(),
            })
            return out
        window_s = min(window_s, MAX_WINDOW_S)

        findings: List[Finding] = []
        detectors = (("dosage", check_dosage), ("pathogen", check_pathogen),
                     ("air", check_air), ("human_action", check_human))
        for key, fn in detectors:
            if key not in working_data:
                continue
            block = working_data[key]
            if not isinstance(block, dict):      # malformed structure: invalid, never crash
                findings.append(Finding(ErrorType.HUMAN, False, "none", 0.0,
                                        detail=f"{key}: expected an object", valid_check=False))
                continue
            findings.append(fn(block, window_s))

        valid_findings = [f for f in findings if f.valid_check]
        if not valid_findings:
            out.update({
                "status": "INSUFFICIENT_OR_INVALID_DATA",
                "confirmed_safe": False,
                "billed_units": 0,
                "latency_ms": ms(),
            })
            return out

        errors = [f for f in valid_findings if f.detected]

        raw_storable = {
            "event_id": event_id,
            "timestamp_utc": timestamp_utc,
            "error_types": [f.kind.value for f in errors],
            "severities": [f.severity for f in errors],
            "device_models": sorted({f.source for f in errors if f.cause == "device" and f.source}),
            "venue_zone": venue.get("zone", ""),
            "window_s": window_s,
        }
        storable = for_storage(
            raw_storable,
            ["event_id", "timestamp_utc", "error_types", "severities",
             "device_models", "venue_zone", "window_s"],
        )

        help_requests = [
            {"finding": f.kind.value, "asked": ask_nearby(nearby_iots, f)}
            for f in errors
        ]
        earners = sorted({a["iot"] for h in help_requests for a in h["asked"]})

        reports = [r for r in (escalate(f, venue, _prevention(f), event_id) for f in errors) if r]
        intentional = [f for f in errors if f.intentional]
        authority = [notify_authorities(f, venue, storable, event_id) for f in intentional]

        verified_emerg = verify_emergency(req, valid_findings)
        billed = acct.settle(
            data_units_btc=1 + len(earners),
            nn_units_bch=2 * max(1, len(errors)),
            succeeded=True,
            verified_emergency=verified_emerg,
        )

        if not errors:
            conservative_certainty = min((f.certainty for f in valid_findings), default=0.0)
            out.update({
                "status": "NO_ERROR",
                "confirmed_safe": True,
                "certainty_score": conservative_certainty,
                "checks_run": [f.kind.value for f in valid_findings],
            })
        else:
            out.update({
                "status": "ERROR_DETECTED",
                "errors": [
                    {
                        "type": f.kind.value,
                        "severity": f.severity,
                        "certainty": f.certainty,
                        "intentional": f.intentional,
                        "cause": f.cause,
                        "source": f.source,
                        "requesting_iot_can_do": f.iot_can_do,
                        "detail": f.detail,
                    }
                    for f in errors
                ],
                "nearby_iots_asked_to_help": help_requests,
                "reports_up_the_chain": reports,
                "authorities_notified": authority or None,
            })

        logger.info(
            "medical respond event=%s iot=%s errors=%d intentional=%d billed=%d %.2fms",
            event_id, acct.iot_id, len(errors), len(intentional), billed, ms(),
        )
        return out

    except Exception as exc:                     # never crash; the finally still purges
        logger.exception("medical respond fallback event=%s iot=%s", event_id, acct.iot_id)
        out.update({"status": "FAILED_SAFE", "reason": type(exc).__name__,
                    "confirmed_safe": False, "billed_units": 0})
        return out

    finally:
        protected_seen = extract_and_purge_protected(working_data)
        out.update({
            "receipt": receipt,
            "stored_monetizable": storable,
            "protected_used_not_stored": protected_seen,
            "iots_earning": earners,
            "billed_units": billed,
            "latency_ms": ms(),
        })


# ---------------------------------------------------------------------------
# 6 · Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    acct = Account("bedside-monitor-12")
    venue = {
        "name": "Example Medical Center",
        "zone": "PACU",
        "device_makers": {"InfuColt-X": "InfuColt Inc."},
    }
    nearby = [
        {"name": "smart infusion pump", "can_help_with": ["dosage"]},
        {"name": "room air unit", "can_help_with": ["air"]},
        {"name": "UV sanitizer", "can_help_with": ["pathogen"]},
    ]

    data = {
        "dosage": {
            "drug": "opioid",
            "administered": 10.0,
            "unit": "mg",
            "safe_low": 1.0,
            "safe_high": 4.0,
            "by_pump": True,
            "pump_model": "InfuColt-X",
            "sensor_conf": 0.95,
            "sources": 2,
            "name": "REDACTED-PATIENT",
            "mrn": "REDACTED",
        },
        "air": {
            "pm2_5": 60,
            "pm2_5_limit": 35,
            "co2_ppm": 800,
            "co2_ppm_limit": 1000,
            "zone": "PACU",
            "sensor_conf": 0.8,
        },
    }
    req = {
        "waiver": True,
        "emergency": True,
        "emergency_attestation_token": "signed-venue-token-001",
        "window_s": 0.8,
    }

    print(json.dumps(respond(acct, req, data, nearby, venue), indent=2, default=str))
