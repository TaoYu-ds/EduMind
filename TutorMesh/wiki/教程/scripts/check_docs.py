#!/usr/bin/env python3
"""检查教程链接和代码语法；--run-offline 仅运行三个已有的纯本地实验。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "wiki" / "教程"
BLOCK = re.compile(r"^```([^\n`]*)\n(.*?)^```\s*$", re.M | re.S)
HEREDOC = re.compile(
    r"(?:python(?:3(?:\.12)?)?)\s+-\s+<<'(?P<tag>[A-Za-z_][A-Za-z0-9_]*)'"
    r"\n(?P<code>.*?)\n(?P=tag)(?=\n|$)", re.S
)
OFFLINE = {
    "01-项目全貌与基础概念.md",
    "05-上下文读取与多轮状态.md",
    "07-Agent路由Skills与工具执行.md",
}


def anchors(text):
    result = set(re.findall(r'<a\s+id="([^"]+)"', text))
    seen = {}
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", BLOCK.sub("", text), re.M):
        title = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = seen.get(title, 0)
        seen[title] = count + 1
        result.add(f"{title}-{count}" if count else title)
    return result


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-offline", action="store_true")
    parser.add_argument("--report", type=Path, help="将检查记录保存为 JSON，不包含 .env")
    args = parser.parse_args()
    errors = []
    counts = dict(documents=0, local_links=0, anchor_links=0, shell_blocks=0,
                  python_programs=0, json_blocks=0, test_selectors=0,
                  offline_experiments=0)
    documents = sorted(DOCS.glob("*.md"))
    for path in documents:
        counts["documents"] += 1
        text = path.read_text(encoding="utf-8")
        label = str(path.relative_to(ROOT))
        prose = BLOCK.sub("", text)
        explicit = re.findall(r'<a\s+id="([^"]+)"', text)
        if len(explicit) != len(set(explicit)):
            errors.append(f"{label}: 有重复的显式锚点")
        for link in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", prose):
            link = link.strip().strip("<>")
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            counts["local_links"] += 1
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.exists():
                errors.append(f"{label}: 链接文件不存在：{link}")
            elif parsed.fragment:
                counts["anchor_links"] += 1
                if target.suffix == ".md" and unquote(parsed.fragment) not in anchors(target.read_text()):
                    errors.append(f"{label}: 锚点不存在：{link}")

        for match in BLOCK.finditer(text):
            language, code = match.group(1).strip(), match.group(2)
            line = text[:match.start()].count("\n") + 1
            try:
                if language in {"bash", "sh"}:
                    counts["shell_blocks"] += 1
                    checked = subprocess.run(["bash", "-n"], input=code, text=True,
                                             capture_output=True, timeout=10)
                    if checked.returncode:
                        errors.append(f"{label}:{line}: Shell 语法：{checked.stderr.strip()}")
                    for program in HEREDOC.finditer(code):
                        counts["python_programs"] += 1
                        ast.parse(program["code"], filename=f"{label}:{line}")
                        if args.run_offline and path.name in OFFLINE:
                            ran = subprocess.run(
                                [sys.executable, "-c", program["code"]], cwd=ROOT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=30,
                            )
                            if ran.returncode:
                                errors.append(f"{label}:{line}: 离线实验失败：{ran.stderr.strip()}")
                            else:
                                counts["offline_experiments"] += 1
                elif language == "json":
                    counts["json_blocks"] += 1
                    json.loads(code)
            except (SyntaxError, ValueError, subprocess.TimeoutExpired) as error:
                errors.append(f"{label}:{line}: {error}")

        for filename, function in re.findall(r"(tests/[\w/.-]+\.py)::(test_\w+)", text):
            counts["test_selectors"] += 1
            target = ROOT / filename
            names = {node.name for node in ast.walk(ast.parse(target.read_text()))
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))} if target.exists() else set()
            if function not in names:
                errors.append(f"{label}: 测试选择器不存在：{filename}::{function}")

    if args.run_offline and counts["offline_experiments"] != len(OFFLINE):
        errors.append("离线实验数量不等于 3，请检查文档中的完整 Python 命令是否被移动或删除")
    source_paths = []
    for folder in ("api", "agents", "core", "memory", "monitor", "tooling", "evaluation", "tests", "skills", "config"):
        source_paths.extend(p for p in (ROOT / folder).rglob("*")
                            if p.is_file() and p.suffix in {".py", ".md", ".json", ".yml", ".yaml", ".conf"})
    source_paths.extend(ROOT / name for name in (
        "Dockerfile", ".dockerignore", ".env.example", "requirements.txt",
        "requirements-dev.txt", "docker-compose.yml", "docker-compose.web.yml"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    report = dict(
        status="failed" if errors else "passed", counts=counts, errors=errors,
        python=platform.python_version(), platform=platform.system(),
        git_head=git.stdout.strip() if git.returncode == 0 else None,
        source_sha256=fingerprint(source_paths),
        tutorial_sha256=fingerprint(documents + list((DOCS / "examples").glob("*")) + [Path(__file__).resolve()]),
        limits="仅本地链接/锚点、Shell/JSON/完整Python语法、测试名称与可选三项离线实验；未运行在线服务或完整pytest。",
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
