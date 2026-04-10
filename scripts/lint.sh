#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# lint.sh — Run shellcheck on all shell scripts in the repo
###############################################################################

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Check for shellcheck
if ! command -v shellcheck &>/dev/null; then
    echo "ERROR: shellcheck not found. Install it first:"
    echo "  macOS: brew install shellcheck"
    echo "  Ubuntu: sudo apt install shellcheck"
    exit 1
fi

echo "→ Running shellcheck…"
echo ""

# Find all .sh files, excluding the ncnn submodule
sh_files=()
while IFS= read -r -d '' f; do
    sh_files+=("$f")
done < <(find "${REPO_ROOT}" -path "${REPO_ROOT}/ncnn" -prune -o -name '*.sh' -print0)

if [[ ${#sh_files[@]} -eq 0 ]]; then
    echo "  No shell scripts found."
    exit 0
fi

echo "  Checking ${#sh_files[@]} file(s):"
for f in "${sh_files[@]}"; do
    rel="${f#"${REPO_ROOT}"/}"
    echo "    - ${rel}"
done
echo ""

# Run shellcheck — exit 1 on any finding (error, warning, info, style)
if shellcheck --color=always --severity=style "${sh_files[@]}"; then
    echo "  ✅ All clear."
else
    echo ""
    echo "  ❌ shellcheck found issues. See above."
    exit 1
fi
