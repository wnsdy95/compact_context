"""
CLI for AILang-IR conversation compression.

Usage:
    python -m ailang_ir compress [-s SPEAKER] [--snippets] [FILE]
    python -m ailang_ir ingest [FILE]
    python -m ailang_ir export [-n N] [--snippets] FILE
    python -m ailang_ir spec
    python -m ailang_ir interactive [--snippets]

Options:
    --snippets          Include condensed source text for detail preservation
    --act-labels        Include stance labels (default: on)
    --no-act-labels     Disable stance labels

FILE defaults to stdin when omitted (except for export).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ailang_ir.pipeline import Pipeline
from ailang_ir.models.domain import SpeakerRole
from ailang_ir.llm.format_spec import get_format_spec
from ailang_ir.llm.validator import validate_code
from ailang_ir.memory.store import MemoryStore


SPEAKER_MAP = {
    "user": SpeakerRole.USER, "u": SpeakerRole.USER,
    "system": SpeakerRole.SYSTEM, "s": SpeakerRole.SYSTEM,
    "agent": SpeakerRole.AGENT, "a": SpeakerRole.AGENT,
}


def _read_input(file_arg: str | None) -> str:
    """Read from file or stdin."""
    if file_arg and file_arg != "-":
        return Path(file_arg).read_text(encoding="utf-8")
    return sys.stdin.read()


def _make_pipeline(store_path: str | None = None) -> Pipeline:
    """Create a pipeline, optionally loading existing memory."""
    pipe = Pipeline()
    if store_path:
        p = Path(store_path)
        if p.exists():
            pipe.memory, ct = MemoryStore.load(p)
            if ct is not None:
                pipe.concept_table = ct
    return pipe


def _save_pipeline(pipe: Pipeline, store_path: str) -> None:
    """Save pipeline memory to disk."""
    pipe.memory.save(Path(store_path), pipe.concept_table)


def cmd_compress(args: argparse.Namespace) -> None:
    """Compress natural language text to semantic codes."""
    text = _read_input(args.file)
    speaker = SPEAKER_MAP.get(args.speaker.lower(), SpeakerRole.USER)

    pipe = _make_pipeline(args.store)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results = []
    for line in lines:
        # Detect speaker prefix (e.g., "User: ...", "Assistant: ...")
        effective_speaker = speaker
        for prefix, role in [("user:", SpeakerRole.USER), ("assistant:", SpeakerRole.AGENT),
                             ("system:", SpeakerRole.SYSTEM)]:
            if line.lower().startswith(prefix):
                effective_speaker = role
                line = line[len(prefix):].strip()
                break

        for r in pipe.process_multi(line, effective_speaker):
            results.append(r)

    # Output
    use_labels = args.act_labels
    use_snippets = args.snippets
    if args.format == "llm":
        from ailang_ir.llm.codec import LLMCodec
        codec = LLMCodec()
        for r in results:
            print(codec.encode(r.frame, act_labels=use_labels, source_snippet=use_snippets))
    elif args.format == "v3":
        for r in results:
            print(r.compact_code)
    elif args.format == "both":
        from ailang_ir.llm.codec import LLMCodec
        codec = LLMCodec()
        for r in results:
            llm_code = codec.encode(r.frame, act_labels=use_labels, source_snippet=use_snippets)
            print(f"{llm_code}  # v3: {r.compact_code}")
    elif args.format == "json":
        out = []
        from ailang_ir.llm.codec import LLMCodec
        codec = LLMCodec()
        for r in results:
            out.append({
                "llm": codec.encode(r.frame, act_labels=use_labels, source_snippet=use_snippets),
                "v3": r.compact_code,
                "summary": r.summary,
            })
        print(json.dumps(out, indent=2, ensure_ascii=False))

    # Stats
    raw_size = len(text.encode("utf-8"))
    if args.format in ("llm", "both"):
        from ailang_ir.llm.codec import LLMCodec
        codec = LLMCodec()
        compressed = "\n".join(codec.encode(r.frame, act_labels=use_labels, source_snippet=use_snippets) for r in results)
    else:
        compressed = "\n".join(r.compact_code for r in results)
    comp_size = len(compressed.encode("utf-8"))
    ratio = (comp_size / raw_size * 100) if raw_size > 0 else 0
    print(f"\n--- {len(results)} frames | {raw_size}B -> {comp_size}B | {ratio:.1f}% ---",
          file=sys.stderr)

    if args.store:
        _save_pipeline(pipe, args.store)
        print(f"Memory saved: {args.store} ({pipe.memory_size} entries)", file=sys.stderr)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest LLM-produced codes into memory."""
    text = _read_input(args.file)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    pipe = _make_pipeline(args.store)

    ok = 0
    errors = 0
    for line in lines:
        # Skip comments
        if line.startswith("#"):
            continue
        vr = validate_code(line)
        if not vr.is_valid:
            print(f"INVALID: {line}", file=sys.stderr)
            for e in vr.errors:
                print(f"  - {e}", file=sys.stderr)
            errors += 1
            continue
        pipe.ingest_code(line)
        ok += 1

    print(f"Ingested: {ok} | Errors: {errors} | Total memory: {pipe.memory_size}",
          file=sys.stderr)

    if args.store:
        _save_pipeline(pipe, args.store)
        print(f"Memory saved: {args.store}", file=sys.stderr)


def cmd_export(args: argparse.Namespace) -> None:
    """Export stored memories as LLM format."""
    if not args.file:
        print("Error: export requires a store file path", file=sys.stderr)
        sys.exit(1)

    pipe = _make_pipeline(args.file)
    if pipe.memory_size == 0:
        print("No memories found.", file=sys.stderr)
        return

    output = pipe.export_context(args.n, source_snippets=args.snippets)
    print(output)
    print(f"\n--- Exported {min(args.n, pipe.memory_size)} of {pipe.memory_size} memories ---",
          file=sys.stderr)


def cmd_spec(args: argparse.Namespace) -> None:
    """Print LLM format spec."""
    print(get_format_spec())


def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive compression REPL."""
    from ailang_ir.llm.codec import LLMCodec

    pipe = _make_pipeline(args.store)
    codec = LLMCodec()
    speaker = SpeakerRole.USER
    use_snippets = args.snippets

    print("AILang-IR Interactive Compression", file=sys.stderr)
    snippet_status = " [snippets ON]" if use_snippets else ""
    print(f"Commands: /speaker <u|s|a> | /export [n] | /snippets | /stats | /quit{snippet_status}", file=sys.stderr)
    print("---", file=sys.stderr)

    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break

            line = line.strip()
            if not line:
                continue

            # Commands
            if line.startswith("/"):
                parts = line.split()
                cmd = parts[0].lower()
                if cmd in ("/quit", "/q", "/exit"):
                    break
                elif cmd == "/speaker":
                    if len(parts) > 1:
                        speaker = SPEAKER_MAP.get(parts[1].lower(), speaker)
                        print(f"Speaker: {speaker.value}", file=sys.stderr)
                elif cmd == "/export":
                    n = int(parts[1]) if len(parts) > 1 else 10
                    print(pipe.export_context(n, source_snippets=use_snippets))
                elif cmd == "/snippets":
                    use_snippets = not use_snippets
                    print(f"Source snippets: {'ON' if use_snippets else 'OFF'}", file=sys.stderr)
                elif cmd == "/stats":
                    for k, v in pipe.stats().items():
                        print(f"  {k}: {v}", file=sys.stderr)
                elif cmd == "/spec":
                    print(get_format_spec())
                elif cmd == "/ingest":
                    # Ingest the rest of the line as LLM code
                    code = " ".join(parts[1:])
                    if code:
                        try:
                            r = pipe.ingest_code(code)
                            print(f"  -> stored: {r.compact_code}")
                        except ValueError as e:
                            print(f"  ERROR: {e}", file=sys.stderr)
                else:
                    print(f"Unknown command: {cmd}", file=sys.stderr)
                continue

            # Process text
            for r in pipe.process_multi(line, speaker):
                llm_code = codec.encode(r.frame, act_labels=True, source_snippet=use_snippets)
                print(f"  {llm_code}  # {r.summary}")

    except KeyboardInterrupt:
        pass

    if args.store:
        _save_pipeline(pipe, args.store)
        print(f"\nMemory saved: {args.store} ({pipe.memory_size} entries)", file=sys.stderr)

    print(f"\nSession: {pipe.memory_size} memories", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ailang-ir",
        description="AILang-IR: Semantic conversation compression",
    )
    sub = parser.add_subparsers(dest="command")

    # compress
    p_compress = sub.add_parser("compress", help="Compress text to semantic codes")
    p_compress.add_argument("file", nargs="?", help="Input file (default: stdin)")
    p_compress.add_argument("-s", "--speaker", default="user", help="Default speaker (user/system/agent)")
    p_compress.add_argument("-f", "--format", default="llm", choices=["llm", "v3", "both", "json"],
                            help="Output format (default: llm)")
    p_compress.add_argument("--store", help="Memory store file (.json) to load/save")
    p_compress.add_argument("--snippets", action="store_true", help="Include source snippets for detail preservation")
    p_compress.add_argument("--act-labels", action="store_true", default=True, dest="act_labels",
                            help="Include act labels for stance clarity (default: on)")
    p_compress.add_argument("--no-act-labels", action="store_false", dest="act_labels",
                            help="Disable act labels")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest LLM-produced codes")
    p_ingest.add_argument("file", nargs="?", help="Input file with LLM codes (default: stdin)")
    p_ingest.add_argument("--store", help="Memory store file (.json) to load/save")

    # export
    p_export = sub.add_parser("export", help="Export stored memories as LLM format")
    p_export.add_argument("file", help="Memory store file (.json)")
    p_export.add_argument("-n", type=int, default=20, help="Number of memories to export (default: 20)")
    p_export.add_argument("--snippets", action="store_true", help="Include source snippets for detail preservation")

    # spec
    sub.add_parser("spec", help="Print LLM format specification")

    # interactive
    p_interactive = sub.add_parser("interactive", help="Interactive compression REPL")
    p_interactive.add_argument("--store", help="Memory store file (.json) to load/save")
    p_interactive.add_argument("--snippets", action="store_true", help="Enable source snippets by default")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "compress": cmd_compress,
        "ingest": cmd_ingest,
        "export": cmd_export,
        "spec": cmd_spec,
        "interactive": cmd_interactive,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
