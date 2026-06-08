## 2024-06-08 - Use usedforsecurity=False with MD5
**Vulnerability:** Use of hashlib.md5() without usedforsecurity=False causes FIPS-compliance crashes and triggers security scanners.
**Learning:** Python 3.9+ supports usedforsecurity=False for non-cryptographic hashes. Since Keras uses Python >= 3.11, this is safe to use everywhere.
**Prevention:** Always add usedforsecurity=False when using MD5 or SHA1 for non-security purposes (like file integrity checking or hashing tricks) to avoid FIPS issues.
