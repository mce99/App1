#!/usr/bin/env bash
set -euo pipefail

OUT="CODEX_CONTEXT.md"
echo "# CODEX CONTEXT (AUTO-GENERATED)" > "$OUT"
echo "" >> "$OUT"

echo "## Repo" >> "$OUT"
echo "- Generated: $(date -Iseconds)" >> "$OUT"
echo "- Path: $(pwd)" >> "$OUT"
echo "- Git remotes:" >> "$OUT"
git remote -v >> "$OUT" 2>/dev/null || echo "(no git remote found)" >> "$OUT"
echo "" >> "$OUT"

echo "## Git status" >> "$OUT"
git status -sb >> "$OUT" 2>/dev/null || echo "(not a git repo)" >> "$OUT"
echo "" >> "$OUT"

echo "## Top-level tree (depth 3)" >> "$OUT"
if command -v tree >/dev/null 2>&1; then
  tree -L 3 -a -I ".git|node_modules|.venv|venv|dist|build|.next|.cache|__pycache__|coverage" >> "$OUT"
else
  find . -maxdepth 3 -type d \
    ! -path "./.git*" ! -path "./node_modules*" ! -path "./.venv*" ! -path "./venv*" \
    ! -path "./dist*" ! -path "./build*" ! -path "./.next*" ! -path "./.cache*" \
    | sed 's|^\./||' | sort >> "$OUT"
fi
echo "" >> "$OUT"

echo "## Detected stack hints" >> "$OUT"
for f in package.json pnpm-lock.yaml yarn.lock bun.lockb \
         pyproject.toml poetry.lock requirements.txt Pipfile \
         go.mod Cargo.toml pom.xml build.gradle \
         docker-compose.yml Dockerfile \
         Makefile; do
  if [ -f "$f" ]; then echo "- Found: $f" >> "$OUT"; fi
done
echo "" >> "$OUT"

echo "## Package scripts / commands" >> "$OUT"
if [ -f package.json ] && command -v node >/dev/null 2>&1; then
  echo "### package.json scripts" >> "$OUT"
  node -e "const p=require('./package.json'); console.log(Object.entries(p.scripts||{}).map(([k,v])=>`- ${k}: ${v}`).join('\n'))" >> "$OUT"
  echo "" >> "$OUT"
fi

echo "## README excerpts" >> "$OUT"
if [ -f README.md ]; then
  # grab the first ~120 lines as a quick runbook
  sed -n '1,120p' README.md >> "$OUT"
else
  echo "(no README.md found)" >> "$OUT"
fi
echo "" >> "$OUT"

echo "## Last 20 commits (for recent context)" >> "$OUT"
git --no-pager log -n 20 --oneline >> "$OUT" 2>/dev/null || echo "(no git history)" >> "$OUT"
echo "" >> "$OUT"

echo "Wrote $OUT"
