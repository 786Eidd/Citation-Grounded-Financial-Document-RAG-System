"""
Ingestion + chunking + provenance tagging for the citation-grounded financial RAG demo.

Source corpus: FinQA (Chen et al., 2021) dev split — real excerpts from S&P 500
companies' SEC 10-K / 10-Q filings, each paired with a question and a
human-annotated "gold" evidence location (a specific sentence or table row).
That gold-evidence annotation is what makes it possible to measure *real*
retrieval accuracy below, instead of eyeballing it.

Every chunk keeps: which filing it came from (doc_id), its exact location in
that filing (chunk_id — e.g. "text_43" or "table_3"), and its type. That's
the "provenance tagging" the portfolio description promises: every chunk can
be traced back to precisely where in which document it came from.
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_SEED = 42
N_DOCS = 80  # number of distinct filing pages to index


def load_raw():
    with open(DATA_DIR / "finqa_dev.json") as f:
        return json.load(f)


def table_row_to_text(table, row_idx):
    """Render a table row as a natural-language sentence, same style FinQA's
    own annotators used for gold evidence, so citations read like prose
    rather than a raw CSV row."""
    header = table[0]
    row = table[row_idx]
    label = row[0]
    parts = [f"the {label} of {header[j]} is {row[j]}" for j in range(1, len(row))]
    return "the " + label + " of " + " ; the ".join(
        f"{header[j]} is {row[j]}" for j in range(1, len(row))
    ) + " ;"


def build_chunks(examples):
    """Returns (chunks, qa_items).
    chunks: list of dicts {global_id, doc_id, chunk_id, type, text}
    qa_items: list of dicts {question, gold_global_ids, doc_id}
    """
    chunks = []
    qa_items = []
    seen_global_ids = set()

    for ex in examples:
        doc_id = ex["filename"]
        combined_text = ex["pre_text"] + ex["post_text"]

        for i, sent in enumerate(combined_text):
            sent = sent.strip()
            if len(sent) < 8:  # skip stray punctuation-only rows (".", "..")
                continue
            gid = f"{doc_id}::text_{i}"
            if gid in seen_global_ids:
                continue
            seen_global_ids.add(gid)
            chunks.append({
                "global_id": gid,
                "doc_id": doc_id,
                "chunk_id": f"text_{i}",
                "type": "text",
                "text": sent,
            })

        table = ex.get("table") or []
        if len(table) > 1:
            for r in range(1, len(table)):
                gid = f"{doc_id}::table_{r}"
                if gid in seen_global_ids:
                    continue
                seen_global_ids.add(gid)
                try:
                    text = table_row_to_text(table, r)
                except Exception:
                    continue
                chunks.append({
                    "global_id": gid,
                    "doc_id": doc_id,
                    "chunk_id": f"table_{r}",
                    "type": "table_row",
                    "text": text,
                })

        qa = ex.get("qa") or {}
        question = qa.get("question")
        gold_inds = qa.get("gold_inds") or {}
        if question and gold_inds:
            gold_global_ids = [f"{doc_id}::{k}" for k in gold_inds.keys()]
            qa_items.append({
                "question": question,
                "gold_global_ids": gold_global_ids,
                "doc_id": doc_id,
                "answer": qa.get("exe_ans"),
            })

    return chunks, qa_items


def main():
    random.seed(RANDOM_SEED)
    raw = load_raw()

    # Dedupe to one example per source filing page, for a corpus that spans
    # many distinct real companies/years rather than repeating the same page.
    by_filename = {}
    for ex in raw:
        by_filename.setdefault(ex["filename"], ex)
    unique_examples = list(by_filename.values())
    random.shuffle(unique_examples)
    selected = unique_examples[:N_DOCS]

    chunks, qa_items = build_chunks(selected)

    out = {
        "n_source_documents": len(selected),
        "n_chunks": len(chunks),
        "n_qa_items": len(qa_items),
        "chunks": chunks,
        "qa_items": qa_items,
    }
    out_path = DATA_DIR / "corpus.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"Indexed {len(selected)} real SEC filing excerpts")
    print(f"Built {len(chunks)} citable chunks ({sum(1 for c in chunks if c['type']=='text')} text, "
          f"{sum(1 for c in chunks if c['type']=='table_row')} table rows)")
    print(f"Collected {len(qa_items)} question/gold-citation pairs for evaluation")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
