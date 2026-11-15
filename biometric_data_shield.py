import hashlib
import hmac
import json
import math
import time
from typing import Any

from cryptography.fernet import Fernet, InvalidToken


class BiometricDataShield:
    """
    Secures sensitive sensor data (thermal, respiratory, behavioral)
    using access verification, data minimization, and symmetric encryption.

    A key is generated per instance unless one is supplied. To decrypt
    elsewhere, share `session_key` over a secure channel.
    """

    _ALLOWED_ACTIVITY = frozenset({"resting", "light", "moderate", "vigorous", "unknown"})

    # Plausible physiological ranges; readings outside these are dropped (None)
    _THERMAL_RANGE = (25.0, 45.0)      # °C
    _RESPIRATORY_RANGE = (0, 80)       # breaths per minute

    def __init__(self, authorized_token_hash: str, key: bytes | None = None) -> None:
        try:
            if len(authorized_token_hash) != 64:
                raise ValueError
            bytes.fromhex(authorized_token_hash)
        except (TypeError, ValueError):
            raise ValueError("authorized_token_hash must be a 64-character hex SHA-256 digest") from None

        self._authorized_token_hash = authorized_token_hash.lower()
        self._key = key if key is not None else Fernet.generate_key()
        self._cipher = Fernet(self._key)

    @property
    def session_key(self) -> bytes:
        """The Fernet key (url-safe base64), usable directly with Fernet(key)."""
        return self._key

    def verify_requester(self, provided_token: str) -> bool:
        """Constant-time verification of the requesting app."""
        if not isinstance(provided_token, str):
            return False
        token_hash = hashlib.sha256(provided_token.encode()).hexdigest()
        return hmac.compare_digest(token_hash, self._authorized_token_hash)

    @staticmethod
    def _clean_number(value: Any, lo: float, hi: float) -> float | None:
        """Return value as a finite float within [lo, hi], else None."""
        if isinstance(value, bool):  # bool is an int subclass; reject it
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        return num if math.isfinite(num) and lo <= num <= hi else None

    def sanitize_telemetry(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """
        Data minimization:
        - allow-list output fields (identifiers and keystroke data never pass)
        - coarse timestamp (minute bucket)
        - reduced precision on continuous values
        - missing, malformed or out-of-range readings become None
        - constrained categorical values
        """
        if not isinstance(raw_data, dict):
            raise TypeError("raw_data must be a dict")

        thermal = self._clean_number(raw_data.get("thermal_c"), *self._THERMAL_RANGE)
        resp = self._clean_number(raw_data.get("respiratory_rpm"), *self._RESPIRATORY_RANGE)
        activity = raw_data.get("activity_level")

        return {
            "timestamp_bucket": int(time.time()) // 60 * 60,
            "thermal_c": None if thermal is None else round(thermal, 1),
            "respiratory_rpm": None if resp is None else round(resp),
            "activity_level": (
                activity
                if isinstance(activity, str) and activity in self._ALLOWED_ACTIVITY
                else "unknown"
            ),
        }

    def export_secure_payload(self, provided_token: str, raw_sensor_data: dict[str, Any]) -> bytes:
        """Validate, minimize, and encrypt. Raises PermissionError on failure."""
        if not self.verify_requester(provided_token):
            raise PermissionError("Unauthorized access to biometric sensors blocked.")

        clean = self.sanitize_telemetry(raw_sensor_data)
        payload = json.dumps(clean, separators=(",", ":")).encode()
        return self._cipher.encrypt(payload)

    def decrypt_payload(self, encrypted: bytes, ttl: int | None = None) -> dict[str, Any]:
        """
        Decrypt a payload produced with this key.
        If ttl (seconds) is given, payloads older than that are rejected.
        """
        try:
            return json.loads(self._cipher.decrypt(encrypted, ttl=ttl))
        except (InvalidToken, ValueError) as exc:  # JSON/Unicode errors subclass ValueError
            raise ValueError("Decryption or JSON decode failed") from exc


# --- Example Usage ---
if __name__ == "__main__":
    APP_SECRET = "trusted_health_app_v1_key"  # demo only; load real secrets from env/secret store
    stored_hash = hashlib.sha256(APP_SECRET.encode()).hexdigest()

    shield = BiometricDataShield(authorized_token_hash=stored_hash)

    sensor_readings = {
        "user_id": "John_Doe_123",
        "thermal_c": 36.842,
        "respiratory_rpm": 15.7,
        "activity_level": "resting",
        "keystroke_dynamics": [112, 98, 105],
    }

    encrypted = shield.export_secure_payload(APP_SECRET, sensor_readings)
    print("Encrypted:", encrypted)
    print("Decrypted:", shield.decrypt_payload(encrypted, ttl=300))

    # The exported key works with a separate Fernet instance
    print("Via exported key:", json.loads(Fernet(shield.session_key).decrypt(encrypted)))

    try:
        shield.export_secure_payload("malicious_scraper_token", sensor_readings)
    except PermissionError as e:
        print("Security Alert:", e)
