#!/usr/bin/env python3
"""
dgg_rag.py — a self-contained RAG (retrieval) engine for an Obsidian markdown vault.

Built as a proof-of-concept for Destiny's public research vault
(https://publish.obsidian.md/destiny). It answers the core problem from the
"magnitude / trust" video: when someone states a claim with the wrong magnitude,
you want the *exact source and figure* from your own vetted notes in one query —
not a third party's summary you have to trust.

Design goals for this POC:
  - ZERO third-party dependencies. Pure Python 3 stdlib. Runs on any machine.
  - Ingests a folder of Obsidian `.md` notes exactly as they sit on disk.
  - Chunks by heading + paragraph, preserves the note title and file path, and
    EXTRACTS THE CITATIONS in each chunk (URLs + APA-style reference lines) so a
    query returns the receipts, not just prose.
  - BM25 lexical retrieval by default (fast, transparent, no model download).
  - Optional `--backend ollama` semantic retrieval using a local Ollama embedding
    model (nomic-embed-text) — the same self-hosted stack Luis runs in his homelab,
    where a production version of this indexes 1,500+ chunks into a real vector DB.

Usage:
    # 1) point it at a vault (defaults to the bundled sample_vault/)
    python3 dgg_rag.py index --vault sample_vault
    python3 dgg_rag.py query "heavy metals in food magnitude" -k 3

    # one-shot (index if needed, then query):
    python3 dgg_rag.py ask "how many casualties" --vault sample_vault

    # semantic backend via local Ollama (matches homelab production):
    python3 dgg_rag.py index --vault sample_vault --backend ollama
    python3 dgg_rag.py query "who do you trust" --backend ollama

Point `--vault` at an export of the real Obsidian vault and it works unchanged.
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, urllib.request
from collections import Counter, defaultdict

INDEX_FILE = ".dgg_index.json"
URL_RE = re.compile(r'https?://[^\s\)\]\>"]+')
# APA-ish reference line: "Author, A. (2024). ..." — a cheap heuristic for a citation line.
APA_RE = re.compile(r'[A-Z][A-Za-z\-]+,\s+[A-Z]\.[^.\n]*\(\d{4}\)')
WORD_RE = re.compile(r"[A-Za-z0-9']+")
STOP = set("the a an and or of to in on for is are was were be been being with as at by "
           "this that these those it its from into about over under it's i you he she we they "
           "not no do does did has have had will would can could should".split())


# ----------------------------- ingest + chunk -----------------------------

def strip_frontmatter(text: str):
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def clean_wikilinks(text: str) -> str:
    # [[Note|alias]] -> alias ; [[Note]] -> Note  (keep the human-readable target)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text


def extract_citations(chunk: str):
    cites = []
    for u in URL_RE.findall(chunk):
        cites.append(u.rstrip(".,);"))
    for m in APA_RE.findall(chunk):
        cites.append(m.strip())
    # dedupe, preserve order
    seen, out = set(), []
    for c in cites:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def chunk_note(path: str, text: str):
    """Yield (title, heading, body, citations) chunks. One chunk per section/paragraph block."""
    title = os.path.splitext(os.path.basename(path))[0]
    body = clean_wikilinks(strip_frontmatter(text))
    heading = title
    buf = []

    def flush(hd, lines):
        block = "\n".join(lines).strip()
        if len(block) >= 30:
            return {"title": title, "heading": hd, "text": block,
                    "citations": extract_citations(block), "path": path}
        return None

    chunks = []
    for line in body.splitlines():
        h = re.match(r"^(#{1,6})\s+(.*)", line)
        if h:
            c = flush(heading, buf)
            if c: chunks.append(c)
            buf = []
            heading = h.group(2).strip()
        elif line.strip() == "" and buf:
            # paragraph break inside a section -> emit if it's already sizable
            if sum(len(x) for x in buf) > 400:
                c = flush(heading, buf)
                if c: chunks.append(c)
                buf = []
        else:
            buf.append(line)
    c = flush(heading, buf)
    if c: chunks.append(c)
    # Note-level citations: any source anywhere in the note, so a hit on the intro
    # paragraph still surfaces the receipts that live under a later heading.
    note_cites = extract_citations(body)
    for c in chunks:
        c["note_citations"] = note_cites
    return chunks


def walk_vault(vault: str):
    chunks = []
    for root, _, files in os.walk(vault):
        for f in files:
            if f.lower().endswith(".md"):
                p = os.path.join(root, f)
                try:
                    text = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                chunks.extend(chunk_note(p, text))
    return chunks


def tokenize(s: str):
    return [w for w in (t.lower() for t in WORD_RE.findall(s)) if w not in STOP and len(w) > 1]


# ----------------------------- BM25 -----------------------------

class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doclen = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doclen) / self.N) if self.N else 0
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, q_tokens, i):
        s = 0.0
        tf = self.tf[i]
        dl = self.doclen[i] or 1
        for t in q_tokens:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            freq = tf[t]
            s += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, q_tokens, k):
        scored = ((self.score(q_tokens, i), i) for i in range(self.N))
        return sorted((x for x in scored if x[0] > 0), reverse=True)[:k]


# ----------------------------- Ollama embeddings (optional) -----------------------------

def ollama_embed(text: str, host="http://threadripper:11434", model="nomic-embed-text"):
    req = urllib.request.Request(
        f"{host}/api/embeddings",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ----------------------------- commands -----------------------------

def cmd_index(args):
    chunks = walk_vault(args.vault)
    if not chunks:
        sys.exit(f"No .md chunks found under {args.vault!r}")
    payload = {"backend": args.backend, "vault": args.vault, "chunks": chunks}
    if args.backend == "ollama":
        print(f"Embedding {len(chunks)} chunks via Ollama ({args.model}) …", file=sys.stderr)
        for c in chunks:
            c["vec"] = ollama_embed(c["text"], args.host, args.model)
    out = os.path.join(args.vault, INDEX_FILE)
    json.dump(payload, open(out, "w"))
    print(f"Indexed {len(chunks)} chunks from {args.vault} -> {out}  (backend={args.backend})")


def load_index(vault):
    p = os.path.join(vault, INDEX_FILE)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def render(hits, chunks):
    if not hits:
        print("  (no matches)")
        return
    for rank, (score, i) in enumerate(hits, 1):
        c = chunks[i]
        print(f"\n[{rank}] {c['title']}  ›  {c['heading']}   (score {score:.3f})")
        print(f"    file: {c['path']}")
        snippet = re.sub(r"\s+", " ", c["text"]).strip()
        print(f"    {snippet[:280]}{'…' if len(snippet) > 280 else ''}")
        if c["citations"]:
            print("    citations (in this passage):")
            for cite in c["citations"]:
                print(f"      • {cite}")
        elif c.get("note_citations"):
            print("    citations (from this note):")
            for cite in c["note_citations"]:
                print(f"      • {cite}")
        else:
            print("    citations: (none in this note)")


def cmd_query(args):
    idx = load_index(args.vault)
    if idx is None:
        sys.exit(f"No index in {args.vault!r}. Run: python3 dgg_rag.py index --vault {args.vault}")
    chunks = idx["chunks"]
    backend = args.backend or idx.get("backend", "bm25")
    print(f"\nQ: {args.query}    [backend={backend}, vault={args.vault}]")
    if backend == "ollama":
        qv = ollama_embed(args.query, args.host, args.model)
        scored = sorted(((cosine(qv, c["vec"]), i) for i, c in enumerate(chunks)
                         if "vec" in c), reverse=True)[:args.k]
        render(scored, chunks)
    else:
        bm = BM25([tokenize(c["text"] + " " + c["heading"] + " " + c["title"]) for c in chunks])
        render(bm.search(tokenize(args.query), args.k), chunks)


def cmd_ask(args):
    if load_index(args.vault) is None:
        cmd_index(args)
    cmd_query(args)


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval over an Obsidian markdown vault.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "query", "ask"):
        s = sub.add_parser(name)
        if name != "index":
            s.add_argument("query", help="query text")
        s.add_argument("--vault", default="sample_vault")
        s.add_argument("--backend", choices=["bm25", "ollama"], default=None)
        s.add_argument("-k", type=int, default=3)
        s.add_argument("--host", default="http://threadripper:11434")
        s.add_argument("--model", default="nomic-embed-text")
    args = ap.parse_args()
    if args.cmd == "index":
        args.backend = args.backend or "bm25"
        cmd_index(args)
    elif args.cmd == "query":
        cmd_query(args)
    else:
        args.backend = args.backend or "bm25"
        cmd_ask(args)


if __name__ == "__main__":
    main()
