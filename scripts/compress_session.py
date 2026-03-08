"""
Self-integration tool: compress a development session into AILang-IR format.

This is the tool that proves AILang-IR's value — an AI agent compressing
its own conversation history for context window optimization.

Input: conversation text (one line per turn, optionally prefixed with speaker)
Output: compressed context ready for injection into the next session

Usage:
    # From file
    PYTHONPATH=src python scripts/compress_session.py session.txt

    # From stdin
    echo "User: Let's fix the parser.
    Agent: I'll update the vocabulary patterns.
    User: Also add tests." | PYTHONPATH=src python scripts/compress_session.py

    # With snippets (preserves details like numbers)
    PYTHONPATH=src python scripts/compress_session.py --snippets session.txt

    # Output format spec + compressed context (ready for system prompt)
    PYTHONPATH=src python scripts/compress_session.py --with-spec session.txt

    # Save memory for future sessions
    PYTHONPATH=src python scripts/compress_session.py --store memory.json session.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ailang_ir.pipeline import Pipeline
from ailang_ir.models.domain import SpeakerRole
from ailang_ir.llm.format_spec import get_format_spec


SPEAKER_PREFIXES = {
    "user:": SpeakerRole.USER,
    "agent:": SpeakerRole.AGENT,
    "assistant:": SpeakerRole.AGENT,
    "system:": SpeakerRole.SYSTEM,
    "claude:": SpeakerRole.AGENT,
    "human:": SpeakerRole.USER,
}


def parse_turns(text: str) -> list[tuple[SpeakerRole, str]]:
    """Parse conversation text into (speaker, text) turns."""
    turns = []
    current_speaker = SpeakerRole.USER  # default

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect speaker prefix
        detected = False
        for prefix, role in SPEAKER_PREFIXES.items():
            if line.lower().startswith(prefix):
                current_speaker = role
                line = line[len(prefix):].strip()
                detected = True
                break

        # Alternating speaker heuristic: if no prefix and previous was user, assume agent
        if not detected and turns:
            prev_speaker = turns[-1][0]
            if prev_speaker == SpeakerRole.USER:
                current_speaker = SpeakerRole.AGENT
            else:
                current_speaker = SpeakerRole.USER

        if line:
            turns.append((current_speaker, line))

    return turns


def main():
    parser = argparse.ArgumentParser(
        description="Compress a development session into AILang-IR format",
    )
    parser.add_argument("file", nargs="?", help="Input file (default: stdin)")
    parser.add_argument("--snippets", action="store_true",
                        help="Include source snippets for detail preservation")
    parser.add_argument("--with-spec", action="store_true",
                        help="Prepend FORMAT_SPEC (ready for system prompt injection)")
    parser.add_argument("--store", help="Save memory to JSON file for future sessions")
    parser.add_argument("--load", help="Load existing memory before processing")
    parser.add_argument("--stats", action="store_true",
                        help="Print compression statistics to stderr")
    args = parser.parse_args()

    # Read input
    if args.file and args.file != "-":
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("No input.", file=sys.stderr)
        sys.exit(1)

    # Build pipeline
    pipe = Pipeline()
    if args.load:
        from ailang_ir.memory.store import MemoryStore
        p = Path(args.load)
        if p.exists():
            pipe.memory, ct = MemoryStore.load(p)
            if ct is not None:
                pipe.concept_table = ct
            print(f"Loaded {pipe.memory_size} existing memories.", file=sys.stderr)

    # Parse and process
    turns = parse_turns(text)
    for speaker, content in turns:
        pipe.process(content, speaker)

    # Export
    compressed = pipe.export_context(
        n=pipe.memory_size,
        source_snippets=args.snippets,
    )

    # Output
    if args.with_spec:
        print(get_format_spec())
        print()
        print("Compressed session context:")

    print(compressed)

    # Stats
    if args.stats or True:  # always show stats on stderr
        raw_words = len(text.split())
        spec_words = len(get_format_spec().split()) if args.with_spec else 0
        comp_words = len(compressed.split()) + spec_words
        ratio = comp_words / raw_words * 100 if raw_words else 0
        dedup_rate = (1 - pipe.memory_size / len(turns)) * 100 if turns else 0

        print(f"\n--- Session compression ---", file=sys.stderr)
        print(f"  Turns: {len(turns)} → {pipe.memory_size} unique ({dedup_rate:.0f}% dedup)", file=sys.stderr)
        print(f"  Words: {raw_words} → {comp_words} ({ratio:.0f}%)", file=sys.stderr)
        if args.snippets:
            print(f"  Mode: hybrid (with source snippets)", file=sys.stderr)
        else:
            print(f"  Mode: codes only", file=sys.stderr)

    # Save
    if args.store:
        pipe.memory.save(Path(args.store), pipe.concept_table)
        print(f"  Memory saved: {args.store} ({pipe.memory_size} entries)", file=sys.stderr)


if __name__ == "__main__":
    main()
