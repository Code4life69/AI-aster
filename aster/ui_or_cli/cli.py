from __future__ import annotations

import argparse
from pathlib import Path

from aster.config import load_config
from aster.orchestrator import AsterOrchestrator
from aster.ui_or_cli.desktop_app import DesktopApp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aster", description="Local coding orchestrator")
    sub = parser.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="Launch the desktop UI")
    ui.set_defaults(command="ui")

    connect = sub.add_parser("connect-github", help="Initialize git and set the GitHub remote")
    connect.add_argument("--project-root", default=".")
    connect.add_argument("--url", required=True)

    plan = sub.add_parser("plan", help="Collect context and generate a patch plan")
    plan.add_argument("goal", help="Natural-language goal")
    plan.add_argument("--project-root", default=".")
    plan.add_argument("--mode", choices=["api", "browser"], default=None)

    apply = sub.add_parser("apply", help="Generate a plan and optionally apply it")
    apply.add_argument("goal", help="Natural-language goal")
    apply.add_argument("--project-root", default=".")
    apply.add_argument("--mode", choices=["api", "browser"], default=None)
    apply.add_argument("--yes", action="store_true", help="Apply without stopping at preview")

    args = parser.parse_args(argv)
    command = args.command or "ui"
    if command == "ui":
        DesktopApp().run()
        return 0

    project_root = Path(args.project_root).resolve()
    config = load_config(project_root)
    orchestrator = AsterOrchestrator(config)

    if command == "connect-github":
        print(orchestrator.connect_remote(args.url))
        return 0

    result = orchestrator.plan(args.goal, mode=args.mode)
    if result.sync_log:
        print("Git sync:")
        for item in result.sync_log:
            print(f"- {item}")
        print()
    print(result.preview)
    if result.warnings:
        print("\nWarnings:")
        for item in result.warnings:
            print(f"- {item}")
    if command == "apply" and args.yes:
        results = orchestrator.apply(result.plan, dry_run=False)
        print("\nApply results:")
        for item in results:
            print(f"- {item}")
    return 0
