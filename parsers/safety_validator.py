"""Static safety validators for CONTRACT.md Section 16 (fail-closed).

CONTRACT.md Section 16 requires the pipeline to FAIL CLOSED if any configuration contains a
public deployment target, credential collection fields, real payment processors, real brand
names in adversarial templates, crawler submission to public search engines, outreach to
real users, or malware / exploit / exfiltration behaviour, and requires those checks to be
implemented "as static configuration validators and CI tests".

Design decisions worth stating explicitly, because each one is a place where a validator can
look thorough while checking nothing:

1. The seven conditions are NOT re-typed here. They are loaded from
   ``preregistration.yaml -> safety_fail_closed_conditions.fail_closed_on``, which was frozen
   in Step 1 and whose SHA-256 is recorded in ``configs/frozen_params.json``. If the frozen
   list and the rules implemented below ever diverge, :func:`load_fail_closed_conditions`
   raises. A validator that silently covers six of seven conditions is worse than no
   validator, because it produces a passing CI badge.

2. Hostnames are not guessed out of prose. Every domain-bearing field in the generated
   artifacts is schema-declared, so those fields are validated directly with
   :func:`check_domain`. Hostnames that appear inside page text can only appear as full URLs
   (the generator emits nothing else), and those are parsed with a real URL parser. A
   heuristic bare-domain scan is layered on top of that, with file-extension collisions
   (``.md``, ``.py``, ``.sh``, ``.js`` are all real ccTLDs) excluded. The limitation is
   recorded in :data:`SCAN_LIMITATIONS` rather than papered over.

3. This module and its tests necessarily contain real brand and real payment-processor
   strings, because that is what a blocklist is. Scanning them would therefore always fail.
   :data:`SELF_EXEMPT_SUFFIXES` names the exemption explicitly; it is deliberately a short,
   auditable list rather than a wildcard.

4. "Unauthorized login portal" and "unauthorized payment portal" are Section 7 page roles
   that MUST be generated. They are made safe structurally, not editorially: a page may
   describe and link to a login or payment step, but :func:`check_html_structure` rejects any
   page containing ``<form>``, ``<input>``, ``<script>``, ``<iframe>`` or an inline event
   handler. The research object is the link that gets presented, never credential capture.

Run as a script to scan a path:

    uv run python parsers/safety_validator.py registry/authorization_graph.yaml
"""

from __future__ import annotations

import ipaddress
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

import yaml

SAFETY_VALIDATOR_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION_PATH = REPO_ROOT / "preregistration.yaml"

# --------------------------------------------------------------------------------------
# The seven Section 16 fail-closed conditions, as frozen in preregistration.yaml.
# --------------------------------------------------------------------------------------
FAIL_CLOSED_CONDITIONS: tuple[str, ...] = (
    "public_ip_or_public_domain_deployment_target",
    "credential_collection_fields",
    "real_payment_processors",
    "real_brand_names_in_adversarial_templates",
    "crawler_submission_to_public_search_engines",
    "outreach_to_real_users",
    "malware_exploit_delivery_or_data_exfiltration_behavior",
)

# --------------------------------------------------------------------------------------
# Reserved / non-routable name and number spaces used by the controlled world.
# --------------------------------------------------------------------------------------
# RFC 2606 / RFC 6761 reserved TLDs. None of these is delegated in the public DNS root, so a
# generated domain cannot resolve to a real host even if a config escaped the sandbox.
RESERVED_TLDS: frozenset[str] = frozenset({"test", "invalid", "localhost", "example"})

# ITU-T E.164 country code 999 is not assigned to any country, so a number in this range
# cannot route to a real subscriber. The NANP 555-01xx fiction range holds only 100 numbers,
# which is fewer than the entity count, so it is not used.
PHONE_PATTERN = re.compile(r"^\+999-555-0\d{4}$")

# Minimum digits for a token to be treated as a dialable international number rather than a
# pattern fragment. E.164 subscriber numbers reachable across borders carry at least this many.
MIN_DIALABLE_DIGITS = 8

# ISO 3166-1 reserves ZZ for user assignment, so a ZZ-prefixed postal code is not a real one.
POSTAL_CODE_PATTERN = re.compile(r"^ZZ-\d{4}$")

# Deliberately NOT WGS84: a synthetic planar grid cannot be reverse-geocoded to a real place.
COORDINATE_SYSTEM = "AEGIS_SYNTHETIC_GRID_V1"
COORDINATE_GRID_MAX = 10_000

DOMAIN_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
DOMAIN_PATTERN = re.compile(rf"^(?:{DOMAIN_LABEL}\.)+[a-z]{{2,24}}$")

# --------------------------------------------------------------------------------------
# Blocklists. Token-matched with word boundaries, case-insensitively.
# --------------------------------------------------------------------------------------
# Real payment processors and card networks (Section 16: "real payment processors").
REAL_PAYMENT_PROCESSORS: tuple[str, ...] = (
    "stripe", "paypal", "braintree", "adyen", "square up", "squareup", "worldpay",
    "authorize.net", "authorizenet", "checkout.com", "mollie", "klarna", "afterpay",
    "affirm", "razorpay", "payu", "paytm", "alipay", "wechat pay", "wechatpay",
    "unionpay", "visa", "mastercard", "maestro", "american express", "amex",
    "discover card", "jcb", "diners club", "sofort", "ideal payment", "bancontact",
    "venmo", "zelle", "cash app", "cashapp", "revolut", "wise transfer", "transferwise",
    "gocardless", "2checkout", "bluesnap", "paddle.com", "lemon squeezy",
)

# Real brands plausible in the restaurant / hotel / travel / directory space this benchmark
# imitates (Section 16: "real brand names in adversarial templates"), plus the large platform
# brands an attacker template would most plausibly reach for.
REAL_BRAND_NAMES: tuple[str, ...] = (
    # Travel / booking / directory
    "booking.com", "expedia", "airbnb", "opentable", "tripadvisor", "yelp", "agoda",
    "trivago", "priceline", "kayak", "hotels.com", "hostelworld", "vrbo", "orbitz",
    "travelocity", "makemytrip", "ctrip", "trip.com", "qunar", "meituan", "dianping",
    "resy", "sevenrooms", "quandoo", "thefork", "grubhub", "doordash", "ubereats",
    "uber eats", "deliveroo", "just eat", "seamless", "postmates", "instacart",
    "eventbrite", "getyourguide", "viator", "klook",
    # Hotels
    "marriott", "hilton", "hyatt", "radisson", "sheraton", "westin", "ritz-carlton",
    "four seasons", "intercontinental", "holiday inn", "premier inn", "travelodge",
    "accor", "wyndham", "best western", "novotel", "ibis hotel", "shangri-la",
    "mandarin oriental", "banyan tree", "jinjiang",
    # Restaurant / food
    "mcdonald", "starbucks", "subway restaurant", "burger king", "kfc", "domino",
    "pizza hut", "wendy", "chipotle", "dunkin", "taco bell", "nando", "pret a manger",
    "costa coffee", "tim hortons", "luckin", "haidilao", "yum china",
    # Platforms / search / cloud
    "google", "alphabet inc", "bing", "yahoo", "duckduckgo", "baidu", "yandex", "naver",
    "facebook", "instagram", "whatsapp", "wechat", "weibo", "tiktok", "douyin",
    "twitter", "linkedin", "reddit", "youtube", "pinterest", "snapchat", "telegram",
    "microsoft", "amazon", "aws", "azure", "cloudflare", "godaddy", "namecheap",
    "shopify", "wordpress", "wix", "squarespace", "godaddy.com",
    "apple inc", "icloud", "openai", "anthropic", "meta platforms",
)

# Search-engine submission / indexing endpoints and config switches.
# `[_\s-]?` rather than `[_-]?` throughout: these appear both as config keys
# (`submit_to_search_engines`) and as prose ("search console"), and matching only the
# underscored form would let the prose spelling through.
SEARCH_ENGINE_SUBMISSION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("indexnow", r"index[_\s-]?now"),
    ("sitemap_ping", r"sitemap[_\s-]?ping|ping\?sitemap"),
    ("search_console", r"search[_\s-]?console|webmaster[_\s-]?tools"),
    ("submit_to_search_engines", r"submit[_\s-]?to[_\s-]?(?:public[_\s-]?)?search[_\s-]?engines?"),
    ("public_crawler_submission", r"public[_\s-]?(?:crawler|index)[_\s-]?submission"),
    ("robots_allow_public", r"submit[_\s-]?url|url[_\s-]?submission[_\s-]?api"),
)

# Outreach to real users: mail/SMS transports and real-recipient addresses.
OUTREACH_PATTERNS: tuple[tuple[str, str], ...] = (
    ("mailto_link", r"mailto:"),
    ("smtp_transport", r"\bsmtp\b|smtp_host|smtp_server|sendmail\b"),
    ("email_service", r"\bsendgrid\b|\bmailgun\b|\bmailchimp\b|\bpostmark\b|\bses[_-]?send\b"),
    ("sms_service", r"\btwilio\b|\bnexmo\b|\bvonage\b|\bplivo\b|sms[_-]?send|send[_-]?sms"),
    ("push_outreach", r"push[_-]?notification[_-]?send|outreach[_-]?to[_-]?real[_-]?users"),
    ("consumer_mailbox", r"@(?:gmail|googlemail|outlook|hotmail|yahoo|icloud|proton|qq|163)\."),
)

# Malware / exploit delivery / exfiltration. Also catches active content in pages, which is
# how a static "informational" page would smuggle behaviour.
MALICIOUS_BEHAVIOUR_PATTERNS: tuple[tuple[str, str], ...] = (
    ("script_element", r"<\s*script\b"),
    ("iframe_element", r"<\s*iframe\b"),
    ("object_or_embed", r"<\s*(?:object|embed|applet)\b"),
    ("inline_event_handler", r"\son(?:error|load|click|mouseover|submit|focus)\s*="),
    ("javascript_uri", r"javascript\s*:"),
    ("data_html_uri", r"data:text/html"),
    ("eval_call", r"\beval\s*\(|\bnew\s+Function\s*\("),
    ("cookie_access", r"document\s*\.\s*cookie|localStorage\s*\.|sessionStorage\s*\."),
    ("shell_pipe_exec", r"\|\s*(?:ba)?sh\b|curl\s+[^\s|]+\s*\|\s*\w*sh\b"),
    ("powershell", r"powershell(?:\.exe)?\s+-|Invoke-(?:Expression|WebRequest)\b"),
    ("base64_decode_exec", r"base64\s+-d|b64decode|FromBase64String"),
    ("reverse_shell", r"reverse\s+shell|/dev/tcp/|nc\s+-e\b"),
    ("exfiltration", r"exfiltrat|data[_-]?exfil|beacon[_-]?to[_-]?"),
    ("exploit_delivery", r"\bmetasploit\b|\bmeterpreter\b|\bshellcode\b|payload[_-]?dropper"),
    ("credential_dump", r"mimikatz|credential[_-]?dump|/etc/shadow"),
)

# Credential collection. Structural HTML checks plus key-name checks for configs.
CREDENTIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("form_element", r"<\s*form\b"),
    ("input_element", r"<\s*(?:input|textarea|select)\b"),
    ("password_type", r"type\s*=\s*[\"']?password"),
    ("autocomplete_credential", r"autocomplete\s*=\s*[\"']?(?:current|new)-password"),
    ("credential_field_name", (
        r"\b(?:password|passwd|pwd|passphrase|card[_-]?number|cardnumber|"
        r"credit[_-]?card|cvv|cvc|card[_-]?verification|iban|routing[_-]?number|"
        r"account[_-]?number|ssn|social[_-]?security|national[_-]?id|"
        r"otp[_-]?code|one[_-]?time[_-]?(?:code|password)|"
        r"security[_-]?(?:code|answer)|mother[^\n]{0,12}maiden)\b"
    )),
    ("credential_capture_verb", r"(?:collect|harvest|capture|store)[_-]?credentials?"),
)

# Keys in a config mapping that must never appear at all.
FORBIDDEN_CONFIG_KEYS: tuple[str, ...] = (
    "password", "passwd", "pwd", "passphrase", "secret", "api_key", "apikey",
    "access_token", "refresh_token", "private_key", "credential", "credentials",
    "card_number", "cardnumber", "cvv", "cvc", "iban", "ssn",
    "public_deployment_target", "public_dns", "public_domain", "public_ip",
    "submit_to_search_engines", "real_payment_processor", "payment_processor_api",
    "smtp_password", "notify_real_users",
)

# TLDs that are real but collide with common file extensions in this repository. A bare
# ``CONTRACT.md`` must not be read as a Moldovan hostname. Full URLs are still checked, so a
# genuine ``https://example.md`` is caught by the URL tier.
FILE_EXTENSION_TLD_COLLISIONS: frozenset[str] = frozenset({
    "md", "py", "sh", "js", "in", "it", "is", "cc", "co", "tv", "pl", "ai", "io", "so",
    "la", "do", "be", "at", "no", "se", "ch", "de", "id", "ma", "my", "sg", "tt", "as",
    "me", "ml", "cd", "im", "sc", "st", "to", "ca", "us", "am", "fm",
})

# High-confidence real public TLDs used by the heuristic bare-hostname tier.
PUBLIC_TLDS: frozenset[str] = frozenset({
    "com", "net", "org", "info", "biz", "gov", "edu", "mil", "int", "app", "dev", "xyz",
    "site", "online", "shop", "store", "cloud", "tech", "space", "website", "click",
    "link", "live", "world", "today", "news", "blog", "wiki", "email", "top", "vip",
    "uk", "cn", "jp", "kr", "ru", "fr", "es", "nl", "br", "au", "nz", "za", "eu",
    "hk", "tw", "th", "vn", "ph", "mx", "ar", "cl", "ng", "ke", "eg", "ae", "sa",
    "il", "ir", "pk", "bd", "lk", "np", "tr", "ua", "pt", "gr", "cz", "sk", "hu",
    "ro", "bg", "hr", "rs", "si", "ee", "lv", "lt", "fi", "dk", "ie", "lu", "mt",
})

BARE_HOSTNAME_PATTERN = re.compile(
    rf"(?<![\w@./-])((?:{DOMAIN_LABEL}\.){{1,4}}([a-z]{{2,24}}))(?![\w-])"
)
URL_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>)\]]+")
IP_LITERAL_PATTERN = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.])")
TEL_PATTERN = re.compile(r"(?:tel:|\+)[0-9][0-9()\s.\-]{6,20}")

# Files whose whole purpose is to name forbidden strings. Naming them here keeps the
# exemption auditable; a wildcard would quietly disable the scan.
SELF_EXEMPT_SUFFIXES: tuple[str, ...] = (
    "parsers/safety_validator.py",
    "tests/test_safety.py",
)

SCAN_LIMITATIONS: tuple[str, ...] = (
    "Bare hostnames are detected heuristically: a token is treated as a hostname only when "
    "its final label is in PUBLIC_TLDS and not in FILE_EXTENSION_TLD_COLLISIONS. A real "
    "domain under an exotic TLD written without a scheme in free prose can therefore be "
    "missed. Domain-bearing fields in generated artifacts are schema-declared and validated "
    "directly by check_domain(), and full URLs are parsed, so this gap does not apply to any "
    "artifact this project generates.",
    "Telephone detection ignores tokens with fewer than MIN_DIALABLE_DIGITS digits, so a bare "
    "local subscriber number written in free prose without a country code is not flagged. Every "
    "phone-bearing field in a generated artifact is validated directly by check_phone(), so the "
    "gap does not apply to anything this project produces.",
    "The real-brand and real-payment-processor blocklists are finite. They cover the travel, "
    "hospitality, food-delivery, directory, platform and payment brands an adversarial "
    "template in this domain would plausibly use, but they are not exhaustive over all "
    "trademarks worldwide.",
    "A generated name built from the frozen invented lexicon could coincidentally match a "
    "small real business not present in the blocklist. No real domain, phone, address or "
    "coordinate is ever used and nothing is deployed publicly, which bounds the consequence.",
)


class SafetyError(RuntimeError):
    """Raised to fail closed when a Section 16 condition is violated."""


@dataclass(frozen=True)
class SafetyViolation:
    """A single Section 16 rule hit.

    Parameters
    ----------
    condition:
        The Section 16 fail-closed condition, one of :data:`FAIL_CLOSED_CONDITIONS`.
    rule_id:
        Stable identifier of the rule that fired, for regression testing.
    detail:
        Human-readable description, including the offending excerpt.
    location:
        Where the violation was found (file path, YAML path, or page id).
    """

    condition: str
    rule_id: str
    detail: str
    location: str = "<inline>"

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.condition}/{self.rule_id}] {self.location}: {self.detail}"


@dataclass
class SafetyReport:
    """Aggregated result of a scan."""

    violations: list[SafetyViolation] = field(default_factory=list)
    scanned: list[str] = field(default_factory=list)
    exempted: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def extend(self, violations: Iterable[SafetyViolation]) -> None:
        self.violations.extend(violations)

    def by_condition(self) -> dict[str, int]:
        counts = {c: 0 for c in FAIL_CLOSED_CONDITIONS}
        for v in self.violations:
            counts[v.condition] = counts.get(v.condition, 0) + 1
        return counts

    def raise_if_unsafe(self) -> None:
        """Fail closed (CONTRACT.md Section 16)."""
        if self.violations:
            head = "\n".join(f"  - {v}" for v in self.violations[:25])
            more = "" if len(self.violations) <= 25 else f"\n  ... and {len(self.violations) - 25} more"
            raise SafetyError(
                f"CONTRACT.md Section 16 fail-closed: {len(self.violations)} violation(s)"
                f" across {len(self.scanned)} scanned item(s).\n{head}{more}"
            )


# ======================================================================================
# Frozen-source wiring
# ======================================================================================
def load_fail_closed_conditions(path: Path | None = None) -> list[str]:
    """Load the Section 16 conditions from the frozen preregistration.

    Raises
    ------
    SafetyError
        If the frozen list cannot be read, or if it does not match the conditions this
        module implements. Divergence means either the preregistration was amended without
        updating the validator, or a rule family was dropped; both must fail loudly.
    """
    prereg_path = path or PREREGISTRATION_PATH
    try:
        loaded = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
        frozen = list(loaded["safety_fail_closed_conditions"]["fail_closed_on"])
    except Exception as exc:  # noqa: BLE001 - fail closed on any read/parse problem
        raise SafetyError(f"cannot read frozen fail-closed conditions from {prereg_path}: {exc}") from exc

    if sorted(frozen) != sorted(FAIL_CLOSED_CONDITIONS):
        missing = sorted(set(frozen) - set(FAIL_CLOSED_CONDITIONS))
        extra = sorted(set(FAIL_CLOSED_CONDITIONS) - set(frozen))
        raise SafetyError(
            "validator rule families diverge from the frozen preregistration list; "
            f"frozen-but-unimplemented={missing}, implemented-but-not-frozen={extra}"
        )
    return frozen


# ======================================================================================
# Field-level validators
# ======================================================================================
def check_domain(domain: str, location: str = "<domain>") -> list[SafetyViolation]:
    """Validate a generated domain name.

    A domain is safe only if it is syntactically a hostname, sits under a reserved TLD, and
    contains no real brand or payment-processor token in any label.
    """
    out: list[SafetyViolation] = []
    cond = "public_ip_or_public_domain_deployment_target"
    if not isinstance(domain, str) or not domain:
        return [SafetyViolation(cond, "domain_empty", f"empty or non-string domain: {domain!r}", location)]

    lowered = domain.lower()
    if not DOMAIN_PATTERN.match(lowered):
        out.append(SafetyViolation(cond, "domain_syntax", f"not a valid hostname: {domain!r}", location))
        return out

    tld = lowered.rsplit(".", 1)[-1]
    if tld not in RESERVED_TLDS:
        out.append(SafetyViolation(
            cond, "domain_tld_not_reserved",
            f"domain {domain!r} uses TLD .{tld}, which is not one of the reserved TLDs "
            f"{sorted(RESERVED_TLDS)}; a non-reserved TLD is a public deployment target",
            location,
        ))

    readable = lowered.replace("-", " ").replace(".", " ")
    out.extend(_blocklist_hits(readable, location, brand_only=True))
    return out


def check_url(url: str, location: str = "<url>") -> list[SafetyViolation]:
    """Validate a URL: scheme, host, absence of userinfo, absence of forbidden brands."""
    out: list[SafetyViolation] = []
    cond = "public_ip_or_public_domain_deployment_target"
    parts = urlsplit(url)

    if parts.scheme.lower() not in {"http", "https"}:
        out.append(SafetyViolation(cond, "url_scheme", f"non-http(s) scheme in {url!r}", location))
        return out
    if parts.username or parts.password:
        out.append(SafetyViolation(
            "credential_collection_fields", "url_userinfo",
            f"URL carries userinfo credentials: {url!r}", location))

    host = (parts.hostname or "").lower()
    if not host:
        out.append(SafetyViolation(cond, "url_no_host", f"URL has no host: {url!r}", location))
        return out

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        out.extend(check_domain(host, location))
    else:
        out.extend(_check_ip_object(ip, url, location))
    return out


def check_ip_literal(text: str, location: str = "<text>") -> list[SafetyViolation]:
    """Flag every globally routable IP literal in ``text``."""
    out: list[SafetyViolation] = []
    for match in IP_LITERAL_PATTERN.finditer(text):
        candidate = match.group(1)
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        out.extend(_check_ip_object(ip, candidate, location))
    return out


def _check_ip_object(ip: ipaddress._BaseAddress, shown: str, location: str) -> list[SafetyViolation]:
    """Reject any address IANA marks as globally reachable."""
    if getattr(ip, "is_global", False):
        return [SafetyViolation(
            "public_ip_or_public_domain_deployment_target", "ip_globally_routable",
            f"globally routable IP literal {shown!r}; only private, loopback, link-local and "
            "documentation ranges are permitted", location,
        )]
    return []


def check_phone(phone: str, location: str = "<phone>") -> list[SafetyViolation]:
    """Require the unassigned-country-code fictional numbering plan."""
    if not isinstance(phone, str) or not PHONE_PATTERN.match(phone):
        return [SafetyViolation(
            "outreach_to_real_users", "phone_not_fictional",
            f"phone {phone!r} does not match the reserved fictional plan "
            f"{PHONE_PATTERN.pattern} (ITU country code 999 is unassigned and cannot route)",
            location,
        )]
    return []


def check_postal_code(code: str, location: str = "<postal_code>") -> list[SafetyViolation]:
    """Require the ISO 3166 user-assigned ``ZZ-NNNN`` form."""
    if not isinstance(code, str) or not POSTAL_CODE_PATTERN.match(code):
        return [SafetyViolation(
            "public_ip_or_public_domain_deployment_target", "postal_code_not_synthetic",
            f"postal code {code!r} does not match {POSTAL_CODE_PATTERN.pattern}", location,
        )]
    return []


def check_coordinates(coord: dict[str, Any], location: str = "<coordinates>") -> list[SafetyViolation]:
    """Require the declared synthetic planar grid rather than WGS84 latitude/longitude.

    Real latitude/longitude would reverse-geocode to a real place, which is exactly the
    "impersonate a public business" failure Section 5.1 forbids.
    """
    out: list[SafetyViolation] = []
    cond = "public_ip_or_public_domain_deployment_target"
    if not isinstance(coord, dict):
        return [SafetyViolation(cond, "coordinates_not_mapping", f"not a mapping: {coord!r}", location)]

    for wgs_key in ("lat", "latitude", "lon", "lng", "longitude", "geo", "wgs84"):
        if wgs_key in coord:
            out.append(SafetyViolation(
                cond, "coordinates_wgs84_field",
                f"WGS84-style field {wgs_key!r} present; real coordinates can be "
                "reverse-geocoded to a real business", location,
            ))
    if coord.get("grid_system") != COORDINATE_SYSTEM:
        out.append(SafetyViolation(
            cond, "coordinates_wrong_grid",
            f"grid_system={coord.get('grid_system')!r}, expected {COORDINATE_SYSTEM!r}", location,
        ))
    for axis in ("grid_x", "grid_y"):
        value = coord.get(axis)
        if not isinstance(value, (int, float)) or not 0 <= value <= COORDINATE_GRID_MAX:
            out.append(SafetyViolation(
                cond, "coordinates_out_of_grid",
                f"{axis}={value!r} outside [0, {COORDINATE_GRID_MAX}]", location,
            ))
    return out


# ======================================================================================
# Text-level validators
# ======================================================================================
def _word_boundary_pattern(token: str) -> re.Pattern[str]:
    """Compile a token blocklist entry, tolerating dots and spaces inside the token."""
    escaped = re.escape(token).replace(r"\ ", r"[\s_-]+")
    lead = r"(?<![\w])" if token[0].isalnum() else ""
    trail = r"(?![\w])" if token[-1].isalnum() else ""
    return re.compile(lead + escaped + trail, re.IGNORECASE)


_BRAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (t, _word_boundary_pattern(t)) for t in REAL_BRAND_NAMES
)
_PROCESSOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (t, _word_boundary_pattern(t)) for t in REAL_PAYMENT_PROCESSORS
)


def _blocklist_hits(text: str, location: str, brand_only: bool = False) -> list[SafetyViolation]:
    out: list[SafetyViolation] = []
    for token, pattern in _BRAND_PATTERNS:
        if pattern.search(text):
            out.append(SafetyViolation(
                "real_brand_names_in_adversarial_templates", "real_brand_name",
                f"real brand token {token!r} present", location,
            ))
    if brand_only:
        return out
    for token, pattern in _PROCESSOR_PATTERNS:
        if pattern.search(text):
            out.append(SafetyViolation(
                "real_payment_processors", "real_payment_processor",
                f"real payment processor token {token!r} present", location,
            ))
    return out


def _regex_family_hits(
    text: str, location: str, condition: str, patterns: Sequence[tuple[str, str]]
) -> list[SafetyViolation]:
    out: list[SafetyViolation] = []
    for rule_id, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            out.append(SafetyViolation(
                condition, rule_id,
                f"pattern {rule_id!r} matched {match.group(0)[:80]!r}", location,
            ))
    return out


def check_bare_hostnames(text: str, location: str = "<text>") -> list[SafetyViolation]:
    """Heuristic tier: flag bare hostnames whose TLD is a real public TLD.

    See :data:`SCAN_LIMITATIONS` for the documented gap this tier does and does not close.
    """
    out: list[SafetyViolation] = []
    seen: set[str] = set()
    for match in BARE_HOSTNAME_PATTERN.finditer(text):
        host, tld = match.group(1).lower(), match.group(2).lower()
        if host in seen:
            continue
        if tld in RESERVED_TLDS:
            continue
        if host.count(".") == 1 and tld in FILE_EXTENSION_TLD_COLLISIONS:
            continue  # e.g. CONTRACT.md, generator.py -- see FILE_EXTENSION_TLD_COLLISIONS
        if tld not in PUBLIC_TLDS:
            continue
        seen.add(host)
        out.append(SafetyViolation(
            "public_ip_or_public_domain_deployment_target", "bare_public_hostname",
            f"bare hostname {host!r} under real public TLD .{tld}", location,
        ))
    return out


def check_telephone_numbers(text: str, location: str = "<text>") -> list[SafetyViolation]:
    """Flag any telephone-looking string that is not in the fictional plan.

    A token carrying fewer than :data:`MIN_DIALABLE_DIGITS` digits is skipped. Without that
    rule, prose *describing* the numbering plan (``+999-555-0NNNN``) matches the leading
    fragment ``+999-555-0`` and is reported as a real number, so documentation cannot be written
    about the rule the validator enforces. Seven digits with no country code is not a routable
    international number, so nothing reachable is being waved through; the residual gap (a bare
    local number in free prose) is recorded in :data:`SCAN_LIMITATIONS` and is closed for
    generated artifacts by :func:`check_phone`, which validates every phone-bearing field
    directly.
    """
    out: list[SafetyViolation] = []
    for match in TEL_PATTERN.finditer(text):
        raw = match.group(0)
        normalised = raw.replace("tel:", "").strip()
        compact = re.sub(r"[()\s.]", "", normalised)
        if sum(ch.isdigit() for ch in compact) < MIN_DIALABLE_DIGITS:
            continue
        # Accept both the canonical form and its punctuation-stripped variant.
        if PHONE_PATTERN.match(compact) or re.match(r"^\+9995550\d{4}$", compact):
            continue
        out.append(SafetyViolation(
            "outreach_to_real_users", "phone_not_fictional",
            f"telephone-like string {raw!r} is outside the reserved fictional plan", location,
        ))
    return out


def check_html_structure(html: str, location: str = "<page>") -> list[SafetyViolation]:
    """Structural page checks.

    Section 7 requires unauthorized login and payment portal pages. They are made safe by
    construction: a page may describe and link to a login or payment step, but must contain
    no interactive input surface and no active content at all.
    """
    out = _regex_family_hits(html, location, "credential_collection_fields", CREDENTIAL_PATTERNS)
    out += _regex_family_hits(
        html, location, "malware_exploit_delivery_or_data_exfiltration_behavior",
        MALICIOUS_BEHAVIOUR_PATTERNS,
    )
    return out


def validate_text(text: str, location: str = "<text>") -> list[SafetyViolation]:
    """Run every text-level rule family over ``text``."""
    if not isinstance(text, str):
        return []
    out: list[SafetyViolation] = []
    out += _blocklist_hits(text, location)
    out += _regex_family_hits(text, location, "credential_collection_fields", CREDENTIAL_PATTERNS)
    out += _regex_family_hits(
        text, location, "crawler_submission_to_public_search_engines",
        SEARCH_ENGINE_SUBMISSION_PATTERNS,
    )
    out += _regex_family_hits(text, location, "outreach_to_real_users", OUTREACH_PATTERNS)
    out += _regex_family_hits(
        text, location, "malware_exploit_delivery_or_data_exfiltration_behavior",
        MALICIOUS_BEHAVIOUR_PATTERNS,
    )
    out += check_ip_literal(text, location)
    out += check_telephone_numbers(text, location)
    for match in URL_PATTERN.finditer(text):
        out += check_url(match.group(0).rstrip(".,;"), location)
    out += check_bare_hostnames(text, location)
    return _dedupe(out)


def validate_page_html(html: str, location: str = "<page>") -> list[SafetyViolation]:
    """Validate a rendered page: text rules plus structural rules."""
    return _dedupe(validate_text(html, location) + check_html_structure(html, location))


def validate_mapping(obj: Any, location: str = "<config>") -> list[SafetyViolation]:
    """Recursively validate a loaded config structure.

    Both keys and string values are checked. Key names are checked against
    :data:`FORBIDDEN_CONFIG_KEYS` because a key alone (``password:``) is a Section 16
    violation regardless of its value.
    """
    out: list[SafetyViolation] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_str = str(key)
                if key_str.lower() in FORBIDDEN_CONFIG_KEYS:
                    out.append(SafetyViolation(
                        _condition_for_key(key_str.lower()), "forbidden_config_key",
                        f"forbidden configuration key {key_str!r}", f"{location}:{path}",
                    ))
                walk(value, f"{path}.{key_str}" if path else key_str)
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
        elif isinstance(node, str):
            out.extend(validate_text(node, f"{location}:{path}"))

    walk(obj, "")
    return _dedupe(out)


def _condition_for_key(key: str) -> str:
    if key in {"public_deployment_target", "public_dns", "public_domain", "public_ip"}:
        return "public_ip_or_public_domain_deployment_target"
    if key == "submit_to_search_engines":
        return "crawler_submission_to_public_search_engines"
    if key in {"real_payment_processor", "payment_processor_api"}:
        return "real_payment_processors"
    if key == "notify_real_users":
        return "outreach_to_real_users"
    return "credential_collection_fields"


def _dedupe(violations: Iterable[SafetyViolation]) -> list[SafetyViolation]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[SafetyViolation] = []
    for v in violations:
        key = (v.condition, v.rule_id, v.detail, v.location)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


# ======================================================================================
# File / tree scanning
# ======================================================================================
def is_self_exempt(path: Path) -> bool:
    """True for the validator itself and its negative-control tests."""
    posix = path.resolve().as_posix()
    return any(posix.endswith(suffix) for suffix in SELF_EXEMPT_SUFFIXES)


def validate_file(path: Path, report: SafetyReport | None = None) -> SafetyReport:
    """Scan one file. YAML/JSON are parsed and walked; anything else is scanned as text."""
    report = report or SafetyReport()
    path = Path(path)
    if is_self_exempt(path):
        report.exempted.append(str(path))
        return report

    rel = _rel(path)
    report.scanned.append(rel)
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            loaded = yaml.load(text, Loader=_yaml_loader())
        except yaml.YAMLError as exc:
            report.extend([SafetyViolation(
                "public_ip_or_public_domain_deployment_target", "unparseable_yaml",
                f"cannot parse YAML, so it cannot be validated: {exc}", rel)])
            return report
        report.extend(validate_mapping(loaded, rel))
    elif path.suffix.lower() in {".html", ".htm"}:
        report.extend(validate_page_html(text, rel))
    else:
        report.extend(validate_text(text, rel))
    return report


def validate_tree(roots: Iterable[Path], patterns: Sequence[str] = ("**/*",)) -> SafetyReport:
    """Scan every matching file under each root, fail-closed style (collect then raise)."""
    report = SafetyReport()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            validate_file(root, report)
            continue
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    validate_file(path, report)
    return report


def assert_safe(report: SafetyReport) -> SafetyReport:
    """Fail closed on any violation, then return the report for logging."""
    report.raise_if_unsafe()
    return report


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _yaml_loader():  # pragma: no cover - trivial capability probe
    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def main(argv: Sequence[str]) -> int:  # pragma: no cover - CLI
    load_fail_closed_conditions()
    targets = [Path(a) for a in argv[1:]] or [REPO_ROOT / "registry", REPO_ROOT / "configs"]
    report = validate_tree(targets)
    print(f"safety_validator v{SAFETY_VALIDATOR_VERSION}: "
          f"scanned={len(report.scanned)} exempted={len(report.exempted)} "
          f"violations={len(report.violations)}")
    for v in report.violations[:50]:
        print(f"  {v}")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
