#!/usr/bin/env python3
"""CPU regressions for 3x env normalisation and the atomic .env.3n publish.

Adapted from MiaAI-Lab scripts/test-env-normalisation.py (PR #98) for the
FlyCockpit 3x layout: head-side normalisation + ambient guard from start.sh,
plus the two publish helpers (local + ssh) and the write_env key flow.
Stdlib only; no cluster, no docker.
"""
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
SOURCE = START.read_text()


def extract(start: str, end: str) -> str:
    i = SOURCE.index(start)
    return SOURCE[i:SOURCE.index(end, i)]


NORM_BLOCK = extract('_cleanup_dspark_env()', "# DSPARK_API_KEYS ambient guard (end)")
PUBLISH_LOCAL = extract("publish_env_local()", "publish_env_ssh()")
PUBLISH_SSH = extract("publish_env_ssh()", "sync_tree()")
WRITE_ENV = extract("write_env() {", "sync_tree() {")


def run_env(content: bytes, extra: str = "") -> subprocess.CompletedProcess:
    workdir = Path(tempfile.mkdtemp())
    tmpdir = workdir / "tmp"
    tmpdir.mkdir()
    env_file = workdir / ".env"
    env_file.write_bytes(content)
    script = f"""set -euo pipefail
export TMPDIR={shlex.quote(str(tmpdir))}
ENV_FILE={shlex.quote(str(env_file))}
{NORM_BLOCK}
printf 'NODE0_HOST=%q\\nVLLM_PORT=%q\\n' "${{NODE0_HOST:-<unset>}}" "${{VLLM_PORT:-<unset>}}"
{extra}
"""
    env = dict(os.environ)
    env.pop("DSPARK_API_KEYS", None)
    env.pop("VLLM_API_KEY", None)
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    result.operator_bytes = env_file.read_bytes()  # type: ignore[attr-defined]
    result.leftovers = sorted(p.name for p in tmpdir.iterdir())  # type: ignore[attr-defined]
    result.snapshot_mode = None  # type: ignore[attr-defined]
    shutil.rmtree(workdir)
    return result


class EnvNormalisationTest(unittest.TestCase):
    def test_plain_file(self):
        r = run_env(b"NODE0_HOST=10.0.0.51\nVLLM_PORT=8888\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NODE0_HOST=10.0.0.51", r.stdout)
        self.assertEqual(r.leftovers, [])

    def test_bom_crlf_normalised_and_snapshot_private(self):
        r = run_env(
            b"\xef\xbb\xbfNODE0_HOST=10.0.0.51\r\n\r\nVLLM_PORT=8888\r\n",
            'printf "MODE=%s\\n" "$(stat -c %a "$_dspark_env_clean" 2>/dev/null || stat -f %Lp "$_dspark_env_clean")"; cat "$_dspark_env_clean"',
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NODE0_HOST=10.0.0.51", r.stdout)
        self.assertIn("MODE=600", r.stdout)
        self.assertNotIn("\r", r.stdout)
        self.assertNotIn("\ufeff", r.stdout)
        self.assertEqual(r.leftovers, [])

    def test_operator_file_byte_identical(self):
        raw = b"\xef\xbb\xbfNODE0_HOST=10.0.0.51\r\nVLLM_PORT=8888\r\n"
        r = run_env(raw)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.operator_bytes, raw)

    def test_source_failure_cleans_snapshot(self):
        r = run_env(b"NODE0_HOST=(unbalanced\n")
        # bash 5 (sparks, CI): non-zero rc. bash 3.2 (stock macOS): the
        # sourced-parse-error exit path reports 0 even with `|| exit`; the
        # operational guarantee — the script never proceeds past a broken
        # .env and leaks no snapshot — must hold on both.
        self.assertNotIn("NODE0_HOST=", r.stdout, "must not proceed past a broken .env")
        self.assertEqual(r.leftovers, [])
        # bash 3.2 (stock macOS) loses the exit status on sourced-file parse
        # errors; bash 5 (sparks, CI, homebrew) preserves it.
        probe = subprocess.run(["bash", "-c", "echo -n \"$BASH_VERSION\""], capture_output=True, text=True)
        if probe.stdout.startswith("3."):
            return
        self.assertNotEqual(r.returncode, 0)

    def test_cleanup_armed_before_mktemp(self):
        self.assertLess(NORM_BLOCK.index("trap _cleanup_dspark_env EXIT"),
                        NORM_BLOCK.index("mktemp"))

    def test_ambient_guard_rejects_mismatch(self):
        # ambient value that differs from the .env value -> exit 2
        workdir = Path(tempfile.mkdtemp())
        env_file = workdir / ".env"
        env_file.write_bytes(b"NODE0_HOST=10.0.0.51\nDSPARK_API_KEYS=file-key\n")
        script = f"""set -euo pipefail
DSPARK_API_KEYS=ambient-key
ENV_FILE={shlex.quote(str(env_file))}
{NORM_BLOCK}
printf 'should-not-reach\\n'
"""
        env = dict(os.environ)
        env.pop("VLLM_API_KEY", None)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("only in .env", r.stderr)
        self.assertNotIn("should-not-reach", r.stdout)
        shutil.rmtree(workdir)

    def test_ambient_guard_allows_matching(self):
        workdir = Path(tempfile.mkdtemp())
        env_file = workdir / ".env"
        env_file.write_bytes(b"NODE0_HOST=10.0.0.51\nDSPARK_API_KEYS=same-key\n")
        script = f"""set -euo pipefail
DSPARK_API_KEYS=same-key
ENV_FILE={shlex.quote(str(env_file))}
{NORM_BLOCK}
printf 'reached keys=%s\\n' "${{DSPARK_API_KEYS:-<unset>}}"
"""
        env = dict(os.environ)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("keys=same-key", r.stdout)
        shutil.rmtree(workdir)


class LocalPublishTest(unittest.TestCase):
    def run_publish(self, dest_dir: Path):
        dest = dest_dir / ".env.3n"
        script = f"""set -euo pipefail
{PUBLISH_LOCAL}
publish_env_local {shlex.quote(str(dest))}
"""
        env = dict(os.environ)
        return subprocess.run(
            ["bash", "-c", script],
            input="NODE0_HOST=h\nVLLM_API_KEY=secret\n",
            capture_output=True, text=True, env=env, timeout=30,
        ), dest

    def test_atomic_publish_0600_no_leftovers(self):
        workdir = Path(tempfile.mkdtemp())
        r, dest = self.run_publish(workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(dest.read_text(), "NODE0_HOST=h\nVLLM_API_KEY=secret\n")
        mode = stat.S_IMODE(dest.stat().st_mode)
        self.assertEqual(mode, 0o600)
        leftovers = [p.name for p in workdir.iterdir() if p.name != ".env.3n"]
        self.assertEqual(leftovers, [], "no tmp files may remain")

    def test_failed_publish_cannot_truncate_previous_file(self):
        workdir = Path(tempfile.mkdtemp())
        dest = workdir / ".env.3n"
        dest.write_text("OLD=CREDENTIALS\n")
        # make the directory read-only: the tmp write fails, old file survives
        os.chmod(workdir, 0o555)
        try:
            r, _ = self.run_publish(workdir)
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(dest.read_text(), "OLD=CREDENTIALS\n")
        finally:
            os.chmod(workdir, 0o755)
            shutil.rmtree(workdir)


class SshPublishTest(unittest.TestCase):
    def extract_remote_script(self) -> str:
        """Capture the remote payload by stubbing SSH."""
        workdir = Path(tempfile.mkdtemp())
        stub = workdir / "capture-ssh"
        stub.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$2" > '
            + shlex.quote(str(workdir / "captured.txt")) + '\nexit 0\n'
        )
        stub.chmod(0o755)
        script = f"""set -euo pipefail
SSH={shlex.quote(str(stub))}
CAPTURE={shlex.quote(str(workdir / 'captured.txt'))}
{PUBLISH_SSH}
publish_env_ssh worker@host "/remote/dir/.env.3n" < /dev/null
"""
        env = dict(os.environ)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        captured = (workdir / "captured.txt").read_text()
        shutil.rmtree(workdir)
        return captured

    def test_remote_payload_is_atomic_0600(self):
        remote = self.extract_remote_script()
        self.assertIn("umask 077", remote)
        self.assertIn("chmod 600", remote)
        self.assertIn("mv -f", remote)
        self.assertIn("trap _cleanup_remote_env EXIT HUP INT TERM", remote)
        # and the remote payload must reference the interpolated dest
        self.assertIn("/remote/dir/.env.3n", remote)

    def test_remote_payload_behaviour(self):
        remote = self.extract_remote_script()
        # Execute the captured payload standalone: feed it input, verify
        # atomic write + mode + cleanup in a sandbox.
        workdir = Path(tempfile.mkdtemp())
        target = workdir / "envs" / ".env.3n"
        target.parent.mkdir()
        script = remote + '\nprintf "done\\n"\n'
        env = dict(os.environ)
        r = subprocess.run(
            ["bash", "-c", script],
            input="KEY=1\n",
            capture_output=True, text=True, env=env, timeout=30,
            cwd=str(workdir),
        )
        # the payload has _env_final set inside; re-run with dest override
        remote2 = remote.replace("/remote/dir/.env.3n", str(target))
        script2 = remote2 + '\nprintf "done\\n"\n'
        r2 = subprocess.run(
            ["bash", "-c", script2],
            input="KEY=1\n",
            capture_output=True, text=True, env=env, timeout=30,
        )
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertEqual(target.read_text(), "KEY=1\n")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        leftovers = [p.name for p in target.parent.iterdir() if p.name != ".env.3n"]
        self.assertEqual(leftovers, [])
        shutil.rmtree(workdir)


class WriteEnvFlowTest(unittest.TestCase):
    def test_keys_and_new_knobs_flow_to_env_3n(self):
        script = f"""set -euo pipefail
DSPARK_VLLM_IMAGE=img:tag
MASTER_ADDR=10.0.0.51
MASTER_PORT=8888
GLOO_IFACE=enp1s0
EPLB_JSON={{}}
VLLM_API_KEY='k-single'
DSPARK_API_KEYS='k1 k2'
DSPARK_MAX_INFLIGHT_PREFILLS=3
DRAFT_SAMPLE_METHOD=greedy
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1
{WRITE_ENV}
write_env 0 /repo 10.0.0.51 /home/ubuntu
"""
        env = dict(os.environ)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('VLLM_API_KEY="k-single"', r.stdout)
        self.assertIn('DSPARK_API_KEYS="k1 k2"', r.stdout)
        self.assertIn("DSPARK_MAX_INFLIGHT_PREFILLS=3", r.stdout)
        self.assertIn("DRAFT_SAMPLE_METHOD=greedy", r.stdout)
        self.assertIn("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900", r.stdout)
        self.assertIn("DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1", r.stdout)

    def test_defaults_when_unset(self):
        script = f"""set -euo pipefail
DSPARK_VLLM_IMAGE=img:tag
MASTER_ADDR=10.0.0.51
MASTER_PORT=8888
GLOO_IFACE=enp1s0
EPLB_JSON={{}}
{WRITE_ENV}
write_env 0 /repo 10.0.0.51 /home/ubuntu
"""
        env = dict(os.environ)
        for k in ("VLLM_API_KEY", "DSPARK_API_KEYS", "DSPARK_MAX_INFLIGHT_PREFILLS",
                  "DRAFT_SAMPLE_METHOD", "DSPARK_ENABLE_ISSUE31_GPU_HOTFIX"):
            env.pop(k, None)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('VLLM_API_KEY=""', r.stdout)
        self.assertIn('DSPARK_API_KEYS=""', r.stdout)
        self.assertIn("DSPARK_MAX_INFLIGHT_PREFILLS=2", r.stdout)
        self.assertIn("DRAFT_SAMPLE_METHOD=probabilistic", r.stdout)
        self.assertIn("DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=0", r.stdout)

    def test_sync_tree_excludes_env_3n(self):
        self.assertIn("--exclude '.env.3n'", SOURCE)
        self.assertIn("--exclude '.env'", SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
