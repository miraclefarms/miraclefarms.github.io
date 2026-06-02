#!/usr/bin/env python3
"""
Post-build CI script: encrypt body content of locked Jekyll posts.

Run after `jekyll build`, before uploading the artifact.
Reads password from _data/site_secrets.json (copied from private content repo).
For each post with `locked: true`, replaces the #post-body-content HTML in
_site/ with an AES-256-GCM encrypted blob + password-gate UI.
"""

import base64
import json
import os
import re
import sys
from pathlib import Path

try:
    import yaml
    from bs4 import BeautifulSoup
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Run: pip install pyyaml beautifulsoup4 cryptography")
    sys.exit(1)


ITERATIONS = 100_000

GATE_TEMPLATE = """\
<div class="password-gate" id="password-gate">
  <div class="pg-icon">🔒</div>
  <p class="pg-hint" id="pg-hint">%(hint)s</p>
  <form class="pg-form" id="pg-form" onsubmit="return false">
    <input type="password" id="pg-input" placeholder="输入密码" autocomplete="off" autofocus />
    <button type="button" id="pg-submit">解锁</button>
  </form>
  <p class="pg-error" id="pg-error" hidden>密码不正确，请重试。</p>
</div>
<script type="application/json" id="__locked-blob">%(blob)s</script>"""


def derive_key(password: bytes, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS)
    return kdf.derive(password)


def encrypt_content(plaintext: str, password: bytes) -> dict:
    salt = os.urandom(16)
    iv = os.urandom(12)
    key = derive_key(password, salt)
    ct = AESGCM(key).encrypt(iv, plaintext.encode("utf-8"), None)
    return {
        "salt": base64.b64encode(salt).decode(),
        "iv": base64.b64encode(iv).decode(),
        "ct": base64.b64encode(ct).decode(),
    }


def parse_front_matter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        return yaml.safe_load(text[3:end]) or {}
    except Exception:
        return {}


def filename_to_slug(name: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", name).removesuffix(".md")


def site_html_path(post_path: Path, fm: dict) -> Path | None:
    date_val = fm.get("date")
    if date_val is not None:
        date_str = str(date_val)[:10]
    else:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", post_path.name)
        if not m:
            return None
        date_str = m.group(1)

    year, month, day = date_str.split("-")
    slug = fm.get("slug") or filename_to_slug(post_path.name)
    return Path("_site") / "notes" / year / month / day / slug / "index.html"


def main() -> None:
    secrets_path = Path("_data/site_secrets.json")
    if not secrets_path.exists():
        print("No _data/site_secrets.json found — skipping encryption.")
        return

    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    password_str = secrets.get("lock_password", "").strip()
    if not password_str:
        print("lock_password not set in site_secrets.json — skipping encryption.")
        return

    password = password_str.encode("utf-8")
    locked_count = 0

    for post_path in sorted(Path("_posts").glob("*.md")):
        text = post_path.read_text(encoding="utf-8")
        fm = parse_front_matter(text)
        if not fm.get("locked"):
            continue

        html_path = site_html_path(post_path, fm)
        if html_path is None or not html_path.exists():
            print(f"[WARN] {post_path.name}: built HTML not found at {html_path} — skipping.")
            continue

        html = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        body_div = soup.find(id="post-body-content")
        if body_div is None:
            print(f"[WARN] {post_path.name}: #post-body-content not found — skipping.")
            continue

        inner_html = body_div.decode_contents()
        blob = encrypt_content(inner_html, password)

        hint = fm.get("password_hint", "本文已加密，请输入密码继续阅读。")
        gate_html = GATE_TEMPLATE % {"hint": hint, "blob": json.dumps(blob)}

        body_div.clear()
        body_div.append(BeautifulSoup(gate_html, "html.parser"))

        html_path.write_text(str(soup), encoding="utf-8")
        print(f"[OK] Encrypted: {post_path.name}")
        locked_count += 1

    print(f"Encryption complete — {locked_count} post(s) encrypted.")


if __name__ == "__main__":
    main()
