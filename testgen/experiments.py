"""Persisted local mutation experiments against bounded mutmut samples.

LLM output is validated JSON test data. The subprocess runs our trusted pytest
driver, never provider-generated Python, imports, shell commands or paths.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import unified_diff
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock
import time
from uuid import UUID, uuid4
import xml.etree.ElementTree as ET

import httpx
from mutmut import Context, list_mutations, mutate

from app.config import Settings
from testgen.llm_client import LLMError, create_client

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
TARGETS = {
    "indicators": ("test_indicators.py",),
    "signals": ("test_signals_backtest.py",),
    "data": ("test_data.py",),
    "backtest": ("test_signals_backtest.py", "test_trade_reports.py"),
}
FUNCTIONS = {
    "indicators": {
        "sma": ("window",), "ema": ("window",), "rsi": ("window",),
        "macd": ("fast", "slow", "signal"), "bollinger_bands": ("window", "num_std"),
    },
    "signals": {"generate_signals": ("buy_threshold", "sell_threshold"), "target_positions": ()},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Mutation:
    id: str
    line: int
    original: str
    replacement: str
    diff: str
    source: str


def mutation_catalog(source: str, filename: str, limit: int) -> tuple[list[Mutation], int]:
    """Stable, source-spanning sample; remove syntactically invalid or identical variants."""
    baseline = ast.dump(ast.parse(source))
    originals = source.splitlines()
    unique: set[str] = set()
    candidates = []
    for relative in list_mutations(Context(source=source, filename=filename)):
        variant, count = mutate(Context(source=source, mutation_id=relative))
        if count != 1:
            continue
        try:
            variant_ast = ast.dump(ast.parse(variant))
        except SyntaxError:
            continue
        if variant_ast == baseline or variant_ast in unique:
            continue
        unique.add(variant_ast)
        changed = variant.splitlines()
        line = relative.line_number
        candidates.append(Mutation(
            _hash(variant)[:16], line + 1, originals[line],
            changed[line] if line < len(changed) else "",
            "".join(unified_diff(source.splitlines(keepends=True), variant.splitlines(keepends=True),
                                 fromfile=filename, tofile=filename)), variant,
        ))
    if len(candidates) <= limit:
        return candidates, len(candidates)
    indices = [i * (len(candidates) - 1) // (limit - 1) for i in range(limit)] if limit > 1 else [0]
    return [candidates[i] for i in indices], len(candidates)


def _scalar(value) -> bool:
    return value is None or (type(value) is str and len(value) <= 40) or (
        type(value) in (int, float) and abs(value) <= 1e12 and math.isfinite(value))


def _expected(value, depth=0) -> bool:
    if isinstance(value, list):
        return depth < 2 and len(value) <= 100 and all(_expected(x, depth + 1) for x in value)
    return _scalar(value)


def parse_cases(reply: str, target: str) -> tuple[list[dict], int]:
    if len(reply) > 50_000:
        raise ValueError("Model yanıtı çok uzun.")
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", reply.strip(), re.I)
    try:
        data = json.loads(fenced[1] if fenced else reply)
    except (ValueError, RecursionError):
        raise ValueError("Model geçerli bir JSON test listesi döndürmedi.") from None
    if not isinstance(data, list) or not data:
        raise ValueError("Model test listesi döndürmedi.")
    allowed = FUNCTIONS.get(target, {})
    valid = []
    for case in data[:18]:
        if not isinstance(case, dict) or set(case) - {"function", "values", "args", "kwargs", "expected", "raises"}:
            continue
        function, values = case.get("function"), case.get("values")
        if not isinstance(function, str) or function not in allowed:
            continue
        args, kwargs, expected, raises = case.get("args", []), case.get("kwargs", {}), case.get("expected"), case.get("raises")
        names = allowed[function]
        if not isinstance(values, list) or len(values) > 100 or not all(_scalar(x) for x in values):
            continue
        if not isinstance(args, list) or len(args) > len(names) or not all(type(x) in (int, float) and _scalar(x) for x in args):
            continue
        if function == "generate_signals" and args:
            continue
        if not isinstance(kwargs, dict) or set(kwargs) - set(names[len(args):]):
            continue
        if not all(type(x) in (int, float) and _scalar(x) for x in kwargs.values()):
            continue
        if raises not in (None, "ValueError") or (raises is None and ("expected" not in case or not _expected(expected))):
            continue
        valid.append({"function": function, "values": values, "args": args, "kwargs": kwargs,
                      "expected": expected if raises is None else None, "raises": raises})
    return valid, len(data)


def _prompt(source, target, previous, feedback):
    text = (
        f"Write up to 18 independent test cases for prediction.{target}. "
        "Return only a JSON array, never Python or shell. Each object has function, "
        "values (the input Series as an array), optional args and kwargs, and "
        "expected (output Series as an array, NaN as null; DataFrame as row arrays "
        "in declared column order). For errors use raises: \"ValueError\". "
        "Give hand-computed expected values. Cover boundaries and ordinary cases. "
        "Return only new cases if this is a refinement round. Functions: "
        f"{json.dumps(FUNCTIONS[target])}.\nSource:\n{source}"
    )
    if previous:
        last = previous[-1]
        text += f"\nPrevious mutation score: {last['score']:.4f}."
        survivors = [m for m in last["mutants"] if m["status"] == "survived"]
        if feedback == "line":
            text += "\nSurviving changes:\n" + "\n".join(
                f"line {m['line']}: {m['original']} -> {m['replacement']}" for m in survivors[:6])
        elif feedback == "code":
            text += "\nSurviving mutant modules:\n" + "\n\n".join(m["source"] for m in survivors[:3])
    return text


@dataclass
class PytestResult:
    status: str
    passed: list[str]
    failed: list[str]
    detail: str = ""


def run_tests(directory: Path, files: list[str], timeout: float = 20) -> PytestResult:
    report = directory / "pytest-report.xml"
    report.unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "-c", "pytest.ini",
               "--confcutdir=.", "-p", "no:cacheprovider", f"--junitxml={report}", *files]
    env = {"PATH": os.defpath, "HOME": str(directory), "TMPDIR": str(directory),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    if "SystemRoot" in os.environ:
        env["SystemRoot"] = os.environ["SystemRoot"]
    try:
        run = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return PytestResult("timeout", [], [], "Pytest zaman sınırına ulaştı.")
    if not report.is_file() or run.returncode not in (0, 1):
        return PytestResult("error", [], [], (run.stdout + run.stderr)[-2400:])
    passed, failed = [], []
    for case in ET.parse(report).iter("testcase"):
        if case.find("skipped") is not None:
            continue
        name = case.get("name", "")
        (failed if any(case.find(tag) is not None for tag in ("failure", "error")) else passed).append(name)
    if not (passed or failed):
        return PytestResult("error", [], [], "Pytest çalıştırılabilir test bulamadı.")
    return PytestResult("killed" if failed else "survived", passed, failed)


def run_experiment(config: dict, emit, client=None, *, root: Path = ROOT) -> dict:
    target = config["target"]
    relative = Path("prediction", f"{target}.py")
    source = (root / relative).read_text(encoding="utf-8")
    mutants, pool_size = mutation_catalog(source, relative.as_posix(), config["mutant_limit"])
    if not mutants:
        raise ValueError("Hedef modülde çalıştırılabilecek mutant yok.")
    rounds, accepted = [], []
    started = time.monotonic()
    context = {"source_hash": _hash(source), "engine": "mutmut 2.5.1 + pytest",
               "catalog": _hash("".join(m.id for m in mutants))[:16], "pool_size": pool_size,
               "selected_mutants": len(mutants)}
    with TemporaryDirectory(prefix="bist-mutation-") as scratch:
        folder = Path(scratch)
        shutil.copytree(root / "prediction", folder / "prediction",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (folder / "pytest.ini").write_text("[pytest]\npythonpath = .\n", encoding="utf-8")
        mutant_file = folder / relative
        if client is None:
            tests = list(TARGETS[target])
            for name in ["conftest.py", *tests]:
                shutil.copyfile(root / "tests" / "prediction" / name, folder / name)
        else:
            tests = ["test_cases.py"]
            shutil.copyfile(Path(__file__).with_name("case_harness.py"), folder / tests[0])
        for round_index in range(1, config["rounds"] + 1):
            count = valid_count = prompt_tokens = completion_tokens = 0
            resolved_model = None
            validation = []
            emit({"phase": "generate" if client else "validate", "round": round_index, **context})
            if client:
                reply = client.generate(_prompt(source, target, rounds, config["feedback"]),
                                        system="Generate declarative JSON test cases only.")
                prompt_tokens, completion_tokens = reply.prompt_tokens, reply.completion_tokens
                resolved_model = reply.model
                candidates, count = parse_cases(reply.text, target)
                emit({"phase": "validate"})
                if candidates:
                    (folder / "cases.json").write_text(json.dumps(candidates), encoding="utf-8")
                    original = run_tests(folder, tests)
                    if original.status not in ("survived", "killed"):
                        raise ValueError(f"Özgün kodda test doğrulaması başarısız: {original.detail}")
                    passed = set(original.passed)
                    for i, candidate in enumerate(candidates):
                        name = f"test_generated_case[case-{i}]"
                        valid = name in passed
                        validation.append({"name": name, "passed": valid, "case": candidate})
                        if valid:
                            valid_count += 1
                            if candidate not in accepted:
                                accepted.append(candidate)
                if not accepted:
                    raise ValueError("Modelin testlerinden hiçbiri özgün kodda geçmedi.")
                (folder / "cases.json").write_text(json.dumps(accepted), encoding="utf-8")
            else:
                original = run_tests(folder, tests)
                if original.status != "survived":
                    raise ValueError(f"Referans testleri özgün kodda geçmedi: {original.detail}")
                count = valid_count = len(original.passed)
                validation = [{"name": name, "passed": True} for name in original.passed]
            emit({"phase": "mutate", "mutants_done": 0})
            results = []
            try:
                for index, mutation in enumerate(mutants, 1):
                    mutant_file.write_text(mutation.source, encoding="utf-8")
                    outcome = run_tests(folder, tests)
                    results.append({**asdict(mutation), "status": outcome.status, "tests": outcome.failed,
                                    "detail": outcome.detail})
                    emit({"mutants_done": index})
            finally:
                mutant_file.write_text(source, encoding="utf-8")
            killed = sum(m["status"] == "killed" for m in results)
            rounds.append({"round": round_index, "score": killed / len(results), "killed": killed,
                           "survived": sum(m["status"] == "survived" for m in results),
                           "timeout": sum(m["status"] == "timeout" for m in results),
                           "errors": sum(m["status"] == "error" for m in results),
                           "generated": count, "valid": valid_count,
                           "suite_size": len(accepted) if client else valid_count,
                           "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                           "resolved_model": resolved_model,
                           "validation": validation,
                           "mutants": results})
            emit({"rounds": rounds, "phase": "feedback"})
            if killed == len(results) or (len(rounds) > 1 and rounds[-1]["score"] <= rounds[-2]["score"]):
                break
    return {**context, "rounds": rounds, "duration_seconds": round(time.monotonic() - started, 3)}


class ExperimentService:
    """One active run; reports are written atomically and can be reloaded by ID."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.directory = settings.experiment_runs_dir
        self.lock = Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mutation")
        self.active: str | None = None

    def _path(self, identifier):
        return self.directory / f"{UUID(str(identifier))}.json"

    def _write(self, run):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(run["id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def get(self, identifier):
        run = json.loads(self._path(identifier).read_text(encoding="utf-8"))
        if run["status"] in ("queued", "running") and run["id"] != self.active:
            return {**run, "status": "failed", "phase": "failed",
                    "error": "Önceki sunucu oturumu deney tamamlanmadan kapandı."}
        return run

    def list(self):
        runs = [self.get(path.stem) for path in self.directory.glob("*.json")]
        return sorted(runs, key=lambda row: row["created_at"], reverse=True)

    def options(self):
        local = False
        try:
            response = httpx.get(self.settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=0.7)
            response.raise_for_status()
            names = {model.get("name") for model in response.json().get("models", [])}
            model = self.settings.ollama_model
            local = model in names or (":" not in model and f"{model}:latest" in names)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            pass
        return {
            "targets": [{"id": key, "name": f"prediction.{key}", "llm": key in FUNCTIONS} for key in TARGETS],
            "providers": [
                {"id": "reference", "name": "Repo referans testleri", "available": True, "reason": None},
                {"id": "openai", "name": self.settings.llm_api_model, "available": bool(self.settings.llm_api_key),
                 "reason": None if self.settings.llm_api_key else "LLM_API_KEY tanımlı değil."},
                {"id": "ollama", "name": self.settings.ollama_model, "available": local,
                 "reason": None if local else "Ollama veya seçili yerel model çalışmıyor."},
            ],
        }

    def start(self, config: dict):
        provider = config["provider"]
        if provider != "reference":
            if config["target"] not in FUNCTIONS:
                raise ValueError("Bu modülde yalnızca repo referans testleri çalıştırılabilir.")
            option = next(x for x in self.options()["providers"] if x["id"] == provider)
            if not option["available"]:
                raise ValueError(option["reason"])
        else:
            config = {**config, "rounds": 1, "feedback": None}
        with self.lock:
            if self.active:
                raise RuntimeError("Bir deney çalışıyor. Tamamlanınca yeniden dene.")
            run = {"id": str(uuid4()), "status": "queued", "phase": "queued", "round": 0, "rounds": [],
                   "created_at": _now(), "config": config, "error": None,
                   "model": {"reference": "Repo referans testleri", "openai": self.settings.llm_api_model,
                             "ollama": self.settings.ollama_model}[provider]}
            self.active = run["id"]
            try:
                self._write(run)
                self.pool.submit(self._execute, run.copy())
            except Exception:
                self.active = None
                raise
            return run

    def _execute(self, run):
        def emit(changes):
            run.update(changes)
            self._write(run)
        try:
            emit({"status": "running"})
            config = run["config"]
            provider = config["provider"]
            client = None if provider == "reference" else create_client(provider, self.settings)
            result = run_experiment(config, emit, client)
            emit({**result, "status": "completed", "phase": "completed", "finished_at": _now()})
        except Exception as exc:
            # Don't persist upstream response bodies; they may contain sensitive provider details.
            if isinstance(exc, LLMError):
                message = "LLM isteği başarısız. Sağlayıcı erişimini, modeli ve API anahtarını kontrol et."
            elif isinstance(exc, ValueError):
                message = str(exc)
            else:
                logger.error("Experiment %s failed (%s)", run["id"], type(exc).__name__)
                message = "Deney çalıştırılamadı. Sunucu günlüklerini kontrol et."
            emit({"status": "failed", "phase": "failed", "error": message, "finished_at": _now()})
        finally:
            with self.lock:
                self.active = None
