#!/usr/bin/env python3
"""Streaming prefill/decode probe against an OpenAI-compatible vLLM endpoint.

Reports TTFT, prefill tok/s, decode-only tok/s (excludes first token), and
usage tokens. One JSON object per line to stdout; human summary to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from statistics import median


def chat_stream(url: str, model: str, prompt: str, max_tokens: int, timeout: int):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    text = []
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                text.append(delta)
    t_end = time.perf_counter()
    return {
        "ttft_s": None if t_first is None else t_first - t0,
        "wall_s": t_end - t0,
        "decode_s": None
        if t_first is None or t_last is None or t_last <= t_first
        else t_last - t_first,
        "text": "".join(text),
        "usage": usage,
    }


def make_prompt(n_tokens: int, nonce: str) -> str:
    # Unique nonce defeats prefix-cache between trials. Filler ~1 tok/word.
    word = "alpha "
    return (
        f"nonce={nonce}\n"
        "After the filler, write integers starting at 1, separated by spaces. "
        "Do not stop and do not add commentary.\n" + (word * n_tokens)
    )


def one_trial(url, model, target_prompt, gen_tokens, timeout, nonce: str):
    prompt = make_prompt(target_prompt, nonce)
    r = chat_stream(url, model, prompt, gen_tokens, timeout)
    u = r["usage"] or {}
    ptok = int(u.get("prompt_tokens") or 0)
    ctok = int(u.get("completion_tokens") or 0)
    ttft = r["ttft_s"]
    decode_s = r["decode_s"]
    prefill = (ptok / ttft) if ttft and ptok else None
    decode = ((ctok - 1) / decode_s) if decode_s and ctok > 1 else None
    wall = (ctok / r["wall_s"]) if r["wall_s"] and ctok else None
    return {
        "target_prompt": target_prompt,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "ttft_s": ttft,
        "wall_s": r["wall_s"],
        "decode_s": decode_s,
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "wall_tok_s": wall,
        "text_head": r["text"][:80],
    }


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="unspecified")
    ap.add_argument("--prompts", default="256,2048,8192")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    sizes = [int(x) for x in args.prompts.split(",") if x.strip()]
    out_fh = sys.stdout if args.out == "-" else open(args.out, "a")
    try:
        for n in sizes:
            rows = []
            for i in range(args.trials):
                try:
                    nonce = f"{args.tag}-p{n}-t{i}-{int(time.time()*1000)}"
                    row = one_trial(
                        args.url, args.model, n, args.gen, args.timeout, nonce
                    )
                    row.update(tag=args.tag, trial=i, ok=True, error=None)
                except Exception as e:
                    row = {
                        "tag": args.tag,
                        "trial": i,
                        "target_prompt": n,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                rows.append(row)
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                print(json.dumps(row), file=sys.stderr)
            ok = [r for r in rows if r.get("ok")]
            summary = {
                "tag": args.tag,
                "kind": "median",
                "target_prompt": n,
                "n_ok": len(ok),
                "prompt_tokens": med(ok, "prompt_tokens"),
                "completion_tokens": med(ok, "completion_tokens"),
                "ttft_s": med(ok, "ttft_s"),
                "prefill_tok_s": med(ok, "prefill_tok_s"),
                "decode_tok_s": med(ok, "decode_tok_s"),
                "wall_tok_s": med(ok, "wall_tok_s"),
            }
            out_fh.write(json.dumps(summary) + "\n")
            out_fh.flush()
            print(
                f"MEDIAN p~{n}: prefill={summary['prefill_tok_s']} "
                f"decode={summary['decode_tok_s']} ttft={summary['ttft_s']}",
                file=sys.stderr,
            )
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == "__main__":
    main()
