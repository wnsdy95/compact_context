"""Tests for the CLI (__main__.py)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_cli(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run ailang_ir CLI as a subprocess."""
    cmd = [sys.executable, "-m", "ailang_ir", *args]
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )


class TestSpec:
    def test_spec_prints_format(self):
        r = run_cli("spec")
        assert r.returncode == 0
        assert "AILang-IR Compressed Format" in r.stdout

    def test_spec_contains_header_layout(self):
        r = run_cli("spec")
        assert "Header" in r.stdout
        assert "Act" in r.stdout


class TestCompress:
    def test_compress_stdin(self):
        r = run_cli("compress", input_text="I think graph memory is a good approach.")
        assert r.returncode == 0
        # Should output at least one LLM format code
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 1
        # LLM code should start with a valid speaker char
        assert lines[0][0] in "USAT?"

    def test_compress_v3_format(self):
        r = run_cli("compress", "-f", "v3", input_text="Graph memory works well.")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 1

    def test_compress_json_format(self):
        r = run_cli("compress", "-f", "json", input_text="Semantic compression is useful.")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "llm" in data[0]
        assert "v3" in data[0]
        assert "summary" in data[0]

    def test_compress_both_format(self):
        r = run_cli("compress", "-f", "both", input_text="I prefer typed models.")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 1
        assert "# v3:" in lines[0]

    def test_compress_multi_line(self):
        text = "I think this is great.\nI also like that approach.\n"
        r = run_cli("compress", input_text=text)
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 2

    def test_compress_with_speaker_prefix(self):
        text = "User: I like graph memory.\nAssistant: Graph memory is efficient."
        r = run_cli("compress", input_text=text)
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 2
        # First should be User, second should be Agent
        assert lines[0][0] == "U"
        assert lines[1][0] == "A"

    def test_compress_speaker_flag(self):
        r = run_cli("compress", "-s", "agent", input_text="The system is running.")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 1
        assert lines[0][0] == "A"

    def test_compress_stats_on_stderr(self):
        r = run_cli("compress", input_text="Graph memory is useful.")
        assert "frames" in r.stderr
        assert "B ->" in r.stderr

    def test_compress_with_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = str(Path(tmpdir) / "mem.json")
            r = run_cli("compress", "--store", store, input_text="I believe in testing.")
            assert r.returncode == 0
            assert Path(store).exists()
            assert "Memory saved" in r.stderr

    def test_compress_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Graph memory is the future.\n")
            f.flush()
            r = run_cli("compress", f.name)
            assert r.returncode == 0
            lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
            assert len(lines) >= 1
        Path(f.name).unlink()


class TestIngest:
    def test_ingest_valid_codes(self):
        codes = "Uoblbn graph_mem\nUandbn sem_compress\n"
        r = run_cli("ingest", input_text=codes)
        assert r.returncode == 0
        assert "Ingested: 2" in r.stderr
        assert "Errors: 0" in r.stderr

    def test_ingest_with_errors(self):
        codes = "Uoblbn graph_mem\nINVALID_CODE\nUandbn sem_compress\n"
        r = run_cli("ingest", input_text=codes)
        assert r.returncode == 0
        assert "Ingested: 2" in r.stderr
        assert "Errors: 1" in r.stderr
        assert "INVALID" in r.stderr

    def test_ingest_skips_comments(self):
        codes = "# This is a comment\nUoblbn graph_mem\n"
        r = run_cli("ingest", input_text=codes)
        assert r.returncode == 0
        assert "Ingested: 1" in r.stderr

    def test_ingest_with_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = str(Path(tmpdir) / "mem.json")
            codes = "Uoblbn graph_mem\nUandbn sem_compress\n"
            r = run_cli("ingest", "--store", store, input_text=codes)
            assert r.returncode == 0
            assert Path(store).exists()


class TestExport:
    def test_export_requires_file(self):
        r = run_cli("export")
        assert r.returncode != 0

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = str(Path(tmpdir) / "mem.json")
            # First compress and store
            run_cli("compress", "--store", store,
                    input_text="Graph memory is efficient.\nI prefer typed models.")
            # Then export
            r = run_cli("export", store)
            assert r.returncode == 0
            lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
            assert len(lines) >= 2
            assert "Exported" in r.stderr

    def test_export_n_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = str(Path(tmpdir) / "mem.json")
            text = "\n".join(f"Sentence number {i} is here." for i in range(5))
            run_cli("compress", "--store", store, input_text=text)
            r = run_cli("export", "-n", "2", store)
            assert r.returncode == 0
            lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
            assert len(lines) <= 2


class TestNoCommand:
    def test_no_command_shows_help(self):
        r = run_cli()
        assert r.returncode == 1


class TestFormatSpecImprovement:
    """Test that the FORMAT_SPEC has boundary visualization."""

    def test_format_spec_has_header_description(self):
        from ailang_ir.llm.format_spec import FORMAT_SPEC
        assert "Header" in FORMAT_SPEC
        assert "Speaker" in FORMAT_SPEC

    def test_format_spec_has_act_labels(self):
        from ailang_ir.llm.format_spec import FORMAT_SPEC
        assert "#act_label" in FORMAT_SPEC
        assert "#disagree" in FORMAT_SPEC

    def test_act_codes_listed(self):
        from ailang_ir.llm.format_spec import FORMAT_SPEC
        assert "Act(2ch)" in FORMAT_SPEC
        assert "bl=believe" in FORMAT_SPEC
