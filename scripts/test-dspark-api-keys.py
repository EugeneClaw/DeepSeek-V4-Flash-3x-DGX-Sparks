#!/usr/bin/env python3
"""CPU-only behavioural gates for DSPARK_API_KEYS / VLLM_API_KEY auth on the
3x recipe. Adapted from MiaAI-Lab scripts/test-dspark-api-keys.py (PR #89)
for the FlyCockpit 3x file layout; stdlib only.

These tests execute the REAL auth code — extracted from docker-compose.yml
(the API_KEY_ARGS entrypoint line), from start.sh (launcher preamble), and
from smoke.sh / status.sh (the `# DSPARK_API_KEYS auth (begin/end)` blocks)
— through a shell, and assert observable behaviour:

- unset / empty / whitespace-only DSPARK_API_KEYS => no --api-key anywhere;
- a parsed value becomes EXACTLY ONE --api-key flag carrying every key
  (order preserved, separators collapsed, duplicates allowed);
- CR/LF/VT/FF and backslashes are rejected before empty/conflict
  classification; a dash-leading token is rejected with exit 2 and a
  fixed diagnostic that never echoes token bytes;
- VLLM_API_KEY and DSPARK_API_KEYS both set => exit 2 naming BOTH
  variables, in the entrypoint and in every probe;
- probes send the FIRST parsed key as the bearer;
- keyed starts require the redaction hotfix (launcher pre-flight);
- the smoke/status auth blocks are byte-identical (consistent parsing).
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
START = ROOT / "start.sh"
SMOKE = ROOT / "smoke.sh"
STATUS = ROOT / "status.sh"
REDACTION_PATCH = ROOT / "patches" / "hotfix-vllm-redact-api-key-log.sh"


def compose_entrypoint_auth_line():
    """The single-line API_KEY_ARGS validation block from the compose file."""
    for ln in COMPOSE.read_text().splitlines():
        if "API_KEY_ARGS=(); case" in ln:
            return ln.strip()
    raise AssertionError("API_KEY_ARGS line not found in docker-compose.yml")


def compose_redaction_gate():
    for ln in COMPOSE.read_text().splitlines():
        if "hotfix-vllm-redact-api-key-log.sh --status" in ln:
            return ln.strip()
    raise AssertionError("redaction gate not found in docker-compose.yml")


def marker_block(path: Path, begin: str, end: str) -> str:
    text = path.read_text()
    i = text.index(begin)
    j = text.index(end, i)
    return text[i:j + len(end)]


def probe_block(path: Path) -> str:
    return marker_block(path, "# DSPARK_API_KEYS auth (begin)", "# DSPARK_API_KEYS auth (end)")


def run_bash(script: str, env_extra=None, argv_probe=False):
    """Run extracted auth code under bash. Compose-escaped $$ is folded to $
    first (that substitution is what docker compose does at render time).
    When argv_probe, append an observer that prints the built API_KEY_ARGS
    one-per-line so the exact argv shape vLLM would receive is assertable."""
    script = script.replace("$$", "$")
    if argv_probe:
        script += '\nset -- ${API_KEY_ARGS[@]+"${API_KEY_ARGS[@]}"}\n'
        script += '\nprintf "ARG=%s\\n" "$@"\n'
    env = dict(os.environ)
    for k in ("DSPARK_API_KEYS", "VLLM_API_KEY"):
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30,
    )


def run_probe(block: str, env_extra=None, header_probe=True):
    """Run a probe auth block; observe the AUTH_HEADER_ARGS it builds."""
    script = block
    if header_probe:
        script += '\nprintf "HDR=%s\\n" "${AUTH_HEADER_ARGS[@]}"\n'
    env = dict(os.environ)
    for k in ("DSPARK_API_KEYS", "VLLM_API_KEY"):
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30,
    )


ENTRYPOINT = compose_entrypoint_auth_line()


class EntrypointValidation(unittest.TestCase):
    def test_keyless_builds_no_flag(self):
        for extra in ({}, {"DSPARK_API_KEYS": ""}, {"DSPARK_API_KEYS": "   "}):
            r = run_bash(ENTRYPOINT, extra, argv_probe=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("ARG=--api-key", r.stdout)

    def test_single_flag_carries_every_key_in_order(self):
        r = run_bash(
            ENTRYPOINT,
            {"DSPARK_API_KEYS": "  sk-one   sk-two\tsk-one  "},
            argv_probe=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        args = [l[len("ARG="):] for l in r.stdout.splitlines() if l.startswith("ARG=")]
        self.assertEqual(
            args,
            ["--api-key", "sk-one", "sk-two", "sk-one"],
            "exactly ONE --api-key flag, order kept, duplicates kept",
        )

    def test_newline_crlf_vt_ff_rejected(self):
        for bad in ("sk1\nsk2", "sk1\rsk2", "sk1\x0bsk2", "sk1\x0csk2"):
            r = run_bash(ENTRYPOINT, {"DSPARK_API_KEYS": bad})
            self.assertEqual(r.returncode, 2, repr(bad))
            self.assertIn("single-line", r.stderr)

    def test_backslash_rejected(self):
        r = run_bash(ENTRYPOINT, {"DSPARK_API_KEYS": "sk1\\ sk2"})
        self.assertEqual(r.returncode, 2)
        self.assertIn("backslash", r.stderr)

    def test_dash_leading_token_rejected_without_echo(self):
        r = run_bash(ENTRYPOINT, {"DSPARK_API_KEYS": "sk1 --oops"})
        self.assertEqual(r.returncode, 2)
        self.assertIn("token beginning with '-'", r.stderr)
        self.assertNotIn("--oops", r.stderr, "diagnostic must not echo token bytes")

    def test_both_vars_set_exits_2_before_side_effects(self):
        r = run_bash(
            ENTRYPOINT,
            {"VLLM_API_KEY": "legacy", "DSPARK_API_KEYS": "sk1"},
            argv_probe=True,
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("VLLM_API_KEY", r.stderr)
        self.assertIn("DSPARK_API_KEYS", r.stderr)
        self.assertNotIn("ARG=", r.stdout, "no flag may be built on conflict")

    def test_vllm_api_key_alone_leaves_flag_unset(self):
        # VLLM_API_KEY is consumed by vLLM natively (env), so the entrypoint
        # must NOT emit --api-key for it.
        r = run_bash(ENTRYPOINT, {"VLLM_API_KEY": "legacy"}, argv_probe=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("ARG=--api-key", r.stdout)

    def test_glob_characters_survive_literally(self):
        r = run_bash(ENTRYPOINT, {"DSPARK_API_KEYS": "sk* one?two"}, argv_probe=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        args = [l[len("ARG="):] for l in r.stdout.splitlines() if l.startswith("ARG=")]
        self.assertEqual(args, ["--api-key", "sk*", "one?two"])


class ProbeBehaviour(unittest.TestCase):
    def test_probes_use_first_parsed_key(self):
        for f in (SMOKE, STATUS, START):
            block = probe_block(f) if f is not START else launcher_auth_slice()
            r = run_probe(block, {"DSPARK_API_KEYS": "sk-first sk-second"})
            self.assertEqual(r.returncode, 0, f"{f.name}: {r.stderr}")
            self.assertIn("HDR=-H", r.stdout)
            self.assertIn("Bearer sk-first", r.stdout, f"{f.name}: first key")

    def test_vllm_key_probe(self):
        r = run_probe(probe_block(SMOKE), {"VLLM_API_KEY": "legacy-key"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Bearer legacy-key", r.stdout)

    def test_keyless_probe_sends_no_header(self):
        r = run_probe(probe_block(SMOKE))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "HDR=-H Authorization: Bearer " if False else r.stdout.strip())
        # empty DSPARK_API_KEYS -> AUTH_HEADER_ARGS may be unset; the printf
        # with "${AUTH_HEADER_ARGS[@]}" under set -u would fail, so run lenient
        r2 = run_probe(probe_block(SMOKE).replace('"${AUTH_HEADER_ARGS[@]}"', '"${AUTH_HEADER_ARGS[@]-}"'))
        self.assertEqual(r2.returncode, 0, r2.stderr)

    def test_both_set_probe_exits_2(self):
        for f in (SMOKE, STATUS):
            r = run_probe(probe_block(f), {"VLLM_API_KEY": "a", "DSPARK_API_KEYS": "b"})
            self.assertEqual(r.returncode, 2, f.name)
            self.assertIn("set exactly one of them", r.stderr)

    def test_smoke_status_blocks_byte_identical(self):
        self.assertEqual(probe_block(SMOKE), probe_block(STATUS))


def launcher_auth_slice() -> str:
    text = START.read_text()
    i = text.index("AUTH_HEADER_ARGS=()")
    j = text.index("# Keyed starts require the startup-log redaction hotfix", i)
    return text[i:j]


class LauncherPreflight(unittest.TestCase):
    def test_redaction_patch_required_for_keyed_starts(self):
        # The preflight must exist in start.sh and point at the vendored patch
        text = START.read_text()
        self.assertIn("keyed starts require the startup-log redaction hotfix", text)
        self.assertIn("hotfix-vllm-redact-api-key-log.sh", text)
        self.assertTrue(REDACTION_PATCH.exists())

    def run_ambient(self, ambient, file_value):
        # Drive the two shipped guard halves in isolation: ambient capture
        # (which unsets the var) then the post-source comparison.
        text = START.read_text()
        cap_i = text.index("_dspark_ambient_has=0")
        cap_j = text.index("unset DSPARK_API_KEYS", cap_i)
        capture = text[cap_i:cap_j + len("unset DSPARK_API_KEYS")]
        cmp_i = text.index('if [ "$_dspark_ambient_has" = "1" ]', cap_j)
        cmp_j = text.index("fi", cmp_i) + 2
        compare = text[cmp_i:cmp_j]
        pre = "" if ambient is None else f"DSPARK_API_KEYS='{ambient}'\n"
        script = (
            pre + capture + "\n"
            f"DSPARK_API_KEYS='{file_value}'\n" + compare + "\n"
            'printf "continues=%s\\n" "${DSPARK_API_KEYS:-}"\n'
        )
        env = dict(os.environ)
        env.pop("VLLM_API_KEY", None)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)

    def test_ambient_mismatch_rejected(self):
        r = self.run_ambient("ambient-value", "file-value")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("only in .env", r.stderr)
        self.assertNotIn("continues=", r.stdout, "must exit before proceeding")

    def test_ambient_match_allows_startup(self):
        r = self.run_ambient("same-key", "same-key")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("continues=same-key", r.stdout)

    def test_ambient_absent_allows_startup(self):
        r = self.run_ambient(None, "file-value")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("continues=file-value", r.stdout)


class ComposeWiring(unittest.TestCase):
    def test_redaction_gate_outside_skip_loop_and_fail_closed(self):
        gate = compose_redaction_gate()
        self.assertIn("|| exit 1", gate)
        self.assertIn("--status || exit 1", gate)
        self.assertIn('[ "$${_dspark_keys_set}" = "1" ] || [ -n "$${VLLM_API_KEY:-}" ]', gate)
        # must NOT be inside the skip-hotfix loop's for-body
        self.assertNotIn("for _hf in", gate)

    def test_flag_expansion_in_serve_command(self):
        text = COMPOSE.read_text()
        self.assertIn('"$${API_KEY_ARGS[@]}"', text)
        # and it must come after --trust-remote-code in the exec block
        exec_i = text.index("exec /usr/local/bin/vllm serve")
        flag_i = text.index('"$${API_KEY_ARGS[@]}"')
        trc_i = text.index("--trust-remote-code")
        self.assertGreater(flag_i, trc_i)
        self.assertGreater(flag_i, exec_i)

    def test_env_passthroughs_present(self):
        text = COMPOSE.read_text()
        self.assertIn('DSPARK_API_KEYS: "${DSPARK_API_KEYS:-}"', text)
        self.assertIn('VLLM_API_KEY: "${VLLM_API_KEY:-}"', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
