#!/usr/bin/env bash
# Scripted walk-through of DGG-RAG for an asciinema recording.
# Types each command out, runs it, and pauses so the cast is readable on replay.
set -u
cd "$(dirname "$0")/.."

BOLD=$'\e[1m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'; DIM=$'\e[2m'; RST=$'\e[0m'

say()  { printf '%s\n' "${DIM}# $*${RST}"; sleep 1.4; }
type_cmd() {                       # simulate typing at a prompt
  printf '%s$ %s' "$GREEN" "$RST"
  local c="$1"; for ((i=0; i<${#c}; i++)); do printf '%s' "${c:$i:1}"; sleep 0.018; done
  printf '\n'; sleep 0.5
}
run()  { type_cmd "$1"; eval "$1"; echo; sleep 2.2; }

clear
printf '%s\n' "${BOLD}${CYAN}DGG-RAG — citation-grounded retrieval over an Obsidian vault${RST}"
printf '%s\n\n' "${DIM}github.com/lmrjr/dgg-rag  ·  zero dependencies, pure Python stdlib${RST}"
sleep 1.8

say "Point it at a folder of Obsidian .md notes and build the index."
run "python3 dgg_rag.py index --vault sample_vault"

say "Now ask it the 'magnitude' question — it returns the passage AND the sources."
run "python3 dgg_rag.py query 'heavy metals in food how big is the risk' -k 1"

say "Different topic. Every answer comes back with its receipts attached."
run "python3 dgg_rag.py query 'can you trust experts and institutions' -k 1"

say "Point --vault at an export of the real vault and it runs unchanged."
sleep 1.2
printf '%s\n' "${BOLD}${GREEN}The homework you already did — retrievable, and cited, in one query.${RST}"
sleep 2.5
