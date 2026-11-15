import base64
import binascii
import json
import os
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class ObfuscatedHealthVault:
    """
    Seals health records with AES-256-GCM.

    - Field names are replaced with neutral codes before encryption
    - Every record is padded to one fixed size, so stored blobs all look alike
    - Each record is bound to its patient_id, so blobs can't be swapped between patients
    - A format-version byte is authenticated with the record, leaving room for
      key rotation or format changes later

    In production, keep the master key in an HSM, KMS or secure enclave,
    never in the same database as the data.
    """

    VERSION = 1
    _NONCE_LEN = 12
    _TAG_LEN = 16
    _LEN_PREFIX = 4

    _SCHEMA_MAP = {
        "condition_trigger": "src_0x01",
        "testosterone_level": "bio_0x02",
        "serotonin_5ht_level": "bio_0x03",
        "recovery_phase": "sts_0x04",
    }
    _REVERSE_MAP = {v: k for k, v in _SCHEMA_MAP.items()}

    def __init__(self, master_key: bytes, pad_to_bytes: int = 512) -> None:
        if not isinstance(master_key, bytes) or len(master_key) != 32:
            raise ValueError("Master key must be exactly 32 bytes for AES-256.")
        if not isinstance(pad_to_bytes, int) or pad_to_bytes <= self._LEN_PREFIX:
            raise ValueError(f"pad_to_bytes must be an integer greater than {self._LEN_PREFIX}")
        self._aesgcm = AESGCM(master_key)
        # Fixed per vault: if callers could change it per record, blob sizes would vary and leak
        self._pad_to = pad_to_bytes

    # --- schema ---------------------------------------------------------

    def _obfuscate_schema(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Swap medical field names for neutral codes; reject unknown fields."""
        if not isinstance(raw_data, dict):
            raise TypeError("health_data must be a dict")
        unknown = raw_data.keys() - self._SCHEMA_MAP.keys()
        if unknown:
            raise ValueError(f"Unmapped fields rejected to prevent schema leakage: {sorted(map(str, unknown))}")
        return {self._SCHEMA_MAP[k]: v for k, v in raw_data.items()}

    def _restore_schema(self, obfuscated: Any) -> dict[str, Any]:
        """Restore field names; anything unexpected means the record is invalid."""
        if not isinstance(obfuscated, dict):
            raise ValueError("Decrypted payload is not a record")
        unknown = obfuscated.keys() - self._REVERSE_MAP.keys()
        if unknown:
            raise ValueError("Decrypted payload contains unknown fields")
        return {self._REVERSE_MAP[k]: v for k, v in obfuscated.items()}

    # --- padding --------------------------------------------------------

    def _pad(self, data: bytes) -> bytes:
        """Length-prefix the data and zero-fill to the fixed block size."""
        room = self._pad_to - self._LEN_PREFIX
        if len(data) > room:
            raise ValueError(f"Record too large for fixed block ({len(data)} > {room}); increase pad_to_bytes")
        return (len(data).to_bytes(self._LEN_PREFIX, "big") + data).ljust(self._pad_to, b"\x00")

    def _unpad(self, padded: bytes) -> bytes:
        """Recover the original bytes using the length prefix."""
        length = int.from_bytes(padded[: self._LEN_PREFIX], "big")
        end = self._LEN_PREFIX + length
        if len(padded) < self._LEN_PREFIX or end > len(padded):
            raise ValueError("Invalid padded payload")
        return padded[self._LEN_PREFIX : end]

    # --- associated data ------------------------------------------------

    @staticmethod
    def _check_patient_id(patient_id: Any) -> bytes:
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient_id must be a non-empty string")
        return patient_id.encode()

    def _aad(self, version: int, patient_id: bytes) -> bytes:
        # Version is authenticated too, so it can't be altered without detection
        return bytes([version]) + patient_id

    # --- public API -----------------------------------------------------

    def seal_record(self, patient_id: str, health_data: dict[str, Any]) -> str:
        """Obfuscate field names, pad to fixed size, encrypt. Returns a url-safe base64 string."""
        pid = self._check_patient_id(patient_id)
        serialized = json.dumps(
            self._obfuscate_schema(health_data), separators=(",", ":"), allow_nan=False
        ).encode()

        nonce = os.urandom(self._NONCE_LEN)  # fresh 96-bit nonce for every record
        ciphertext = self._aesgcm.encrypt(nonce, self._pad(serialized), self._aad(self.VERSION, pid))
        return base64.urlsafe_b64encode(bytes([self.VERSION]) + nonce + ciphertext).decode()

    def unseal_record(self, patient_id: str, encrypted_blob: str) -> dict[str, Any]:
        """Decrypt, verify, and restore the original record in memory."""
        pid = self._check_patient_id(patient_id)

        try:
            raw = base64.urlsafe_b64decode(encrypted_blob)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise ValueError("Malformed encrypted blob") from exc

        if len(raw) < 1 + self._NONCE_LEN + self._TAG_LEN:
            raise ValueError("Encrypted blob too short")

        version, nonce, ciphertext = raw[0], raw[1 : 1 + self._NONCE_LEN], raw[1 + self._NONCE_LEN :]
        if version != self.VERSION:
            raise ValueError(f"Unsupported record version: {version}")

        try:
            padded = self._aesgcm.decrypt(nonce, ciphertext, self._aad(version, pid))
        except InvalidTag as exc:
            raise ValueError("Decryption failed (wrong key, tampered data, or patient_id mismatch)") from exc

        try:
            payload = json.loads(self._unpad(padded))
        except ValueError as exc:  # covers JSON and Unicode errors
            raise ValueError("Decrypted payload is not valid JSON") from exc

        return self._restore_schema(payload)


# --- Example Usage ---
if __name__ == "__main__":
    # Generate a 256-bit key (in production: keep it in an HSM/KMS)
    secret_key = AESGCM.generate_key(bit_length=256)
    vault = ObfuscatedHealthVault(secret_key)

    anonymous_user_token = "usr_8849201a"
    sensitive_biomarkers = {
        "condition_trigger": "post_viral_acute_phase",
        "testosterone_level": "145_ng_dL_depleted",
        "serotonin_5ht_level": "42_ng_mL_depleted",
        "recovery_phase": "month_3_rebound",
    }

    # What gets saved to the database/cloud
    stored_blob = vault.seal_record(anonymous_user_token, sensitive_biomarkers)
    print(f"Stored in Database:\n{stored_blob}\n")

    # Decrypted only on the authorized user's device
    print(f"Decrypted in Local Memory:\n{vault.unseal_record(anonymous_user_token, stored_blob)}\n")

    # The same blob can't be opened under another patient's ID
    try:
        vault.unseal_record("usr_someone_else", stored_blob)
    except ValueError as e:
        print("Security Alert:", e)
