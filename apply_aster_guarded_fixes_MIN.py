#!/usr/bin/env python3
from pathlib import Path
import argparse
import difflib
import json
import re
import shutil
from datetime import datetime

def read(p): return p.read_text(encoding="utf-8")
def write(p, s): p.write_text(s, encoding="utf-8", newline="\n")

def diff(name, a, b):
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                     fromfile=f"{name} (before)",
                                     tofile=f"{name} (after)", lineterm=""):
        print(line)

def patch_pyproject(t):
    deps = ['"pyautogui"', '"pywinauto"', '"pyperclip"']
    if all(d in t for d in deps):
        return t, "already has runtime deps"
    m = re.search(r'(?ms)^dependencies\s*=\s*\[(.*?)\]', t)
    if not m:
        return t, "dependencies list not found"
    inner = m.group(1).rstrip()
    if inner and not inner.strip().endswith(","):
        inner += ","
    if inner and not inner.endswith("\n"):
        inner += "\n"
    for d in deps:
        if d not in inner:
            inner += f"    {d},\n"
    return t[:m.start()] + f"dependencies = [{inner}]" + t[m.end():], "added runtime deps"

def patch_models(t):
    changed = t
    changed = changed.replace("auto_commit_and_push: bool = True", "auto_commit_and_push: bool = False")
    changed = changed.replace("push_runtime_logs_after_plan: bool = True", "push_runtime_logs_after_plan: bool = False")
    return changed, ("safer defaults applied" if changed != t else "defaults already safe or pattern missing")

def patch_json(t):
    try:
        data = json.loads(t)
    except Exception:
        return t, "json parse failed"
    before = json.dumps(data, sort_keys=True)
    if data.get("auto_commit_and_push") is True:
        data["auto_commit_and_push"] = False
    if data.get("push_runtime_logs_after_plan") is True:
        data["push_runtime_logs_after_plan"] = False
    after = json.dumps(data, sort_keys=True)
    return (json.dumps(data, indent=2) + "\n", "safer json defaults applied") if after != before else (t, "json already safe")

def patch_orchestrator(t):
    new = t.replace('ensure_branch("main")', '# preserve current branch')
    return new, ("removed forced main branch" if new != t else "forced main branch call not found")

def patch_run_ps1(t):
    new = t
    if "pyautogui" not in new:
        new = new.replace('"requests"', '"requests", "pyautogui", "pywinauto", "pyperclip"', 1)
        new = new.replace("'requests'", "'requests', 'pyautogui', 'pywinauto', 'pyperclip'", 1)
    note = []
    note.append("expanded module preflight" if new != t else "preflight already expanded or pattern missing")
    if "RedirectStandardOutput" in new or "RedirectStandardError" in new:
        note.append("launch redirection already present")
    else:
        note.append("manual review recommended for launch redirection")
    return new, "; ".join(note)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=".")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    targets = [
        (repo / "pyproject.toml", patch_pyproject),
        (repo / "aster" / "config" / "models.py", patch_models),
        (repo / "aster.config.example.json", patch_json),
        (repo / "aster" / "orchestrator.py", patch_orchestrator),
        (repo / "run.ps1", patch_run_ps1),
    ]

    print(f"Repo: {repo}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print()

    changes = []
    notes = []

    for path, fn in targets:
        if not path.exists():
            notes.append(f"{path.relative_to(repo)}: file not found")
            continue
        old = read(path)
        new, note = fn(old)
        notes.append(f"{path.relative_to(repo)}: {note}")
        if new != old:
            changes.append((path, old, new))

    if not changes:
        print("No guarded changes matched.\n")
        for n in notes:
            print("-", n)
        return

    print("Planned changes:")
    for path, old, new in changes:
        rel = path.relative_to(repo)
        print(f"- {rel}")
        diff(str(rel), old, new)
        print()

    print("Notes:")
    for n in notes:
        print("-", n)

    if args.dry_run:
        print("\nDry run only. No files changed.")
        return

    backup_root = repo / ".aster_patch_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    for path, old, new in changes:
        dst = backup_root / path.relative_to(repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        write(path, new)

    print(f"\nBackups written to: {backup_root}")
    print("Guarded fixes applied. Review git diff before committing.")

if __name__ == "__main__":
    main()
