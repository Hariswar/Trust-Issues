import re
import sys
import textwrap
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import path_one


# Rule 1: how well each sentence matches the title.
SENTENCE_TITLE_MATCH_THRESHOLD = 0.14
TITLE_MISMATCH_PENALTY = 0.9
TITLE_MISMATCH_LATE_PENALTY = 0.45
TITLE_MISMATCH_FULL_PENALTY_CHUNKS = 1

# Rule 2: sentence citation density threshold.
SENTENCE_CITATION_THRESHOLD = 0.002

# Rule 3: cross-article chunk corroboration.
CORROBORATION_SIMILARITY_THRESHOLD = 0.16
CORROBORATION_TOP_K = 8
CORROBORATION_MIN_SUPPORT_ARTICLES = 2
CORROBORATION_SAMPLE_ARTICLES = 300
CORROBORATION_MAX_CHUNKS_PER_ARTICLE = 6
CORROBORATION_INDEX = None
LOW_CORROBORATION_PENALTY = 0.35
SINGLE_SOURCE_CORROBORATION_BONUS = 0.35
MULTI_SOURCE_CORROBORATION_BONUS = 0.95

# Rule 4: named-entity support in current chunk only.
ENTITY_MENTION_MIN_COUNT = 2
ENTITY_UNSUPPORTED_PENALTY = 0.35
ENTITY_SUPPORTED_BONUS = 0.25

# Chunking setup (Option 1): paragraph-aware packing with overlap.
CHUNK_TARGET_WORDS = 120
CHUNK_OVERLAP_SENTENCES = 1

SOURCE_PHRASES = [
    "according to",
    "reported by",
    "reports",
    "statement",
    "official",
    "data",
    "study",
    "research",
    "survey",
    "evidence",
    "confirmed",
    "announced",
]

EVIDENCE_WORDS = [
    "data",
    "study",
    "research",
    "report",
    "evidence",
    "official",
    "source",
    "court",
    "document",
    "records",
]

HEDGE_WORDS = [
    "allegedly",
    "reportedly",
    "might",
    "may",
    "could",
    "possibly",
    "appears",
    "seems",
]

ABSOLUTE_WORDS = [
    "always",
    "never",
    "everyone",
    "nobody",
    "all",
    "none",
    "guaranteed",
    "proven",
]

ATTRIBUTION_WORDS = [
    "said",
    "says",
    "stated",
    "according to",
    "claimed",
    "told",
    "wrote",
]


def load_test_data():
    test_path = Path(path_one) / "test (1).csv"
    data = pd.read_csv(test_path, sep=";")

    # Match existing preprocessing convention where the first column
    # is a generated index column in this dataset.
    first_col = data.columns[0]
    if first_col not in {"title", "text", "label"}:
        data = data.drop(columns=[first_col])

    return data.dropna(subset=["title", "text"]).reset_index(drop=True)


def split_into_sentences(text):
    # Split article text into short sentence units.
    if not isinstance(text, str):
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    candidates = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    sentences = [s.strip() for s in candidates if len(s.strip()) >= 8]

    # Fallback: if punctuation was removed upstream, split into fixed-size
    # chunks so we still get sentence-level style outputs.
    if len(sentences) <= 1:
        words = normalized.split()
        chunk_size = 28
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if len(chunk) >= 8:
                chunks.append(chunk)
        sentences = chunks

    return sentences


def split_into_chunks(
    text,
    target_words=CHUNK_TARGET_WORDS,
    overlap_sentences=CHUNK_OVERLAP_SENTENCES,
):
    # Break text by paragraph first, then pack neighboring sentences until
    # a word budget is reached. Keep a small overlap to reduce boundary loss.
    if not isinstance(text, str):
        return []

    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()
    ]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current_sentences = []
    current_words = 0

    def append_chunk():
        nonlocal current_sentences, current_words
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text and (not chunks or chunk_text != chunks[-1]):
            chunks.append(chunk_text)

        if overlap_sentences > 0 and current_sentences:
            current_sentences = current_sentences[-overlap_sentences:]
            current_words = sum(len(s.split()) for s in current_sentences)
        else:
            current_sentences = []
            current_words = 0

    for paragraph in paragraphs:
        paragraph_sentences = split_into_sentences(paragraph)
        if not paragraph_sentences:
            continue

        for sentence in paragraph_sentences:
            sentence_words = len(sentence.split())
            if (
                current_sentences
                and current_words + sentence_words > target_words
            ):
                append_chunk()

            current_sentences.append(sentence)
            current_words += sentence_words

        # Prefer paragraph boundaries as natural split points.
        if current_sentences and current_words >= int(target_words * 0.8):
            append_chunk()

    if current_sentences:
        append_chunk()

    return chunks


def sentence_title_similarity(title, sentences):
    if not sentences:
        return np.array([])

    # Fit on title + its own sentences, then compare each sentence to title.
    vectorizer = TfidfVectorizer(stop_words="english", norm="l2")
    corpus = [title] + sentences
    matrix = vectorizer.fit_transform(corpus)

    title_vec = matrix[0]
    sentence_vecs = matrix[1:]

    # Since vectors are L2-normalized, dot product is cosine similarity.
    similarities = sentence_vecs.dot(title_vec.T).toarray().ravel()
    return similarities


def sentence_citation_density(sentences):
    densities = []
    for sentence in sentences:
        lowered = sentence.lower()
        count = 0
        for phrase in SOURCE_PHRASES:
            count += lowered.count(phrase)
        density = count / max(len(lowered), 1)
        densities.append(density)
    return np.array(densities, dtype=float)


def sentence_feature_counts(sentences):
    citation_hits = []
    evidence_hits = []
    hedge_hits = []
    absolute_hits = []
    absolute_in_quotes_hits = []
    absolute_outside_quotes_hits = []
    attribution_hits = []
    has_number = []
    entity_mentions = []

    # Simple, lightweight heuristic for entity-like text spans.
    # Examples matched: "New York", "Alec Baldwin", "U.S.", "NASA"
    entity_pattern = re.compile(
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}(?:\.[A-Z]{1,})*)\b"
    )

    for sentence in sentences:
        lowered = sentence.lower()

        cit = sum(lowered.count(phrase) for phrase in SOURCE_PHRASES)
        evd = sum(lowered.count(word) for word in EVIDENCE_WORDS)
        hed = sum(lowered.count(word) for word in HEDGE_WORDS)
        quote_spans = re.findall(r'"(.*?)"', lowered)
        quoted_text = " ".join(quote_spans)
        outside_text = re.sub(r'".*?"', " ", lowered)

        abs_in_quotes = sum(
            quoted_text.count(word) for word in ABSOLUTE_WORDS
        )
        abs_outside_quotes = sum(
            outside_text.count(word) for word in ABSOLUTE_WORDS
        )
        abs_count = abs_in_quotes + abs_outside_quotes

        attr = sum(lowered.count(word) for word in ATTRIBUTION_WORDS)
        num = 1 if re.search(r"\d", lowered) else 0

        citation_hits.append(cit)
        evidence_hits.append(evd)
        hedge_hits.append(hed)
        absolute_hits.append(abs_count)
        absolute_in_quotes_hits.append(abs_in_quotes)
        absolute_outside_quotes_hits.append(abs_outside_quotes)
        attribution_hits.append(attr)
        has_number.append(num)
        ent = len(entity_pattern.findall(sentence))
        entity_mentions.append(ent)

    return {
        "citation_hits": np.array(citation_hits, dtype=int),
        "evidence_hits": np.array(evidence_hits, dtype=int),
        "hedge_hits": np.array(hedge_hits, dtype=int),
        "absolute_hits": np.array(absolute_hits, dtype=int),
        "absolute_in_quotes_hits": np.array(
            absolute_in_quotes_hits, dtype=int
        ),
        "absolute_outside_quotes_hits": np.array(
            absolute_outside_quotes_hits, dtype=int
        ),
        "attribution_hits": np.array(attribution_hits, dtype=int),
        "has_number": np.array(has_number, dtype=int),
        "entity_mentions": np.array(entity_mentions, dtype=int),
    }


def build_chunk_corroboration_index(
    dataframe,
    sample_index,
    sample_articles=CORROBORATION_SAMPLE_ARTICLES,
    max_chunks_per_article=CORROBORATION_MAX_CHUNKS_PER_ARTICLE,
):
    if dataframe.empty:
        return None

    available = dataframe.drop(index=sample_index, errors="ignore")
    if available.empty:
        return None

    sample_count = min(sample_articles, len(available))
    sampled = available.sample(n=sample_count, random_state=1)

    chunk_texts = []
    article_ids = []

    for row_idx, row in sampled.iterrows():
        chunks = split_into_chunks(row.get("text", ""))[:max_chunks_per_article]
        for chunk in chunks:
            chunk_texts.append(chunk)
            article_ids.append(int(row_idx))

    if not chunk_texts:
        return None

    vectorizer = TfidfVectorizer(stop_words="english", norm="l2")
    matrix = vectorizer.fit_transform(chunk_texts)

    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "article_ids": np.array(article_ids, dtype=int),
    }


def chunk_corroboration_signals(
    chunks,
    corroboration_index,
    top_k=CORROBORATION_TOP_K,
    similarity_threshold=CORROBORATION_SIMILARITY_THRESHOLD,
):
    if not chunks or corroboration_index is None:
        zeros = np.zeros(len(chunks), dtype=float)
        return zeros, zeros.astype(int)

    vectorizer = corroboration_index["vectorizer"]
    matrix = corroboration_index["matrix"]
    article_ids = corroboration_index["article_ids"]

    query_matrix = vectorizer.transform(chunks)
    similarities = query_matrix.dot(matrix.T).toarray()

    max_similarities = np.zeros(len(chunks), dtype=float)
    support_article_counts = np.zeros(len(chunks), dtype=int)

    for i in range(len(chunks)):
        row = similarities[i]
        if row.size == 0:
            continue

        k = min(top_k, row.size)
        top_idx = np.argpartition(row, -k)[-k:]
        top_scores = row[top_idx]

        max_similarities[i] = float(top_scores.max())
        supported_articles = {
            int(article_ids[idx])
            for idx, score in zip(top_idx, top_scores)
            if score >= similarity_threshold
        }
        support_article_counts[i] = len(supported_articles)

    return max_similarities, support_article_counts


def score_sentence(
    similarity,
    citation_density,
    features,
    chunk_position,
    full_penalty_chunks,
):
    # Higher score means more likely false.
    score = 0.0
    reasons = []

    if similarity < SENTENCE_TITLE_MATCH_THRESHOLD:
        if chunk_position <= full_penalty_chunks:
            score += TITLE_MISMATCH_PENALTY
            reasons.append("low title match")
        else:
            score += TITLE_MISMATCH_LATE_PENALTY
            reasons.append("low title match (late chunk reduced)")

    if (
        citation_density < SENTENCE_CITATION_THRESHOLD
        and features["citation_hits"] == 0
    ):
        score += 0.8
        reasons.append("no citation cues")

    if features["hedge_hits"] > 0:
        score += 0.5
        reasons.append("hedging language")

    if features["absolute_outside_quotes_hits"] > 0:
        score += 0.6
        reasons.append("absolute claim words outside quotes")

    if features["absolute_in_quotes_hits"] > 0:
        score += 0.2
        reasons.append("absolute claim words inside quotes")

    if features["absolute_in_quotes_hits"] > 0 and features[
        "attribution_hits"
    ] > 0:
        score -= 0.2
        reasons.append("quoted claim appears attributed")

    if features["evidence_hits"] > 0:
        score -= 0.6
        reasons.append("contains evidence words")

    if features["citation_hits"] > 0:
        score -= 0.4
        reasons.append("contains citation phrase")

    if features["entity_mentions"] >= ENTITY_MENTION_MIN_COUNT:
        if features["evidence_hits"] > 0 or features["citation_hits"] > 0:
            score -= ENTITY_SUPPORTED_BONUS
            reasons.append("named entities have support cues")
        else:
            score += ENTITY_UNSUPPORTED_PENALTY
            reasons.append("named entities without support cues")

    if features["max_chunk_similarity"] < CORROBORATION_SIMILARITY_THRESHOLD:
        score += LOW_CORROBORATION_PENALTY
        reasons.append("low cross-article corroboration")

    if features["support_articles"] >= 1:
        score -= SINGLE_SOURCE_CORROBORATION_BONUS
        reasons.append("corroborated by another article")

    if features["support_articles"] >= CORROBORATION_MIN_SUPPORT_ARTICLES:
        score -= MULTI_SOURCE_CORROBORATION_BONUS
        reasons.append("corroborated by multiple articles")

    score = float(np.clip(score, 0.0, 2.0))
    if not reasons:
        reasons.append("neutral signal")

    return score, reasons


def label_from_score(score):
    if score >= 1.2:
        return "likely_false"
    if score >= 0.6:
        return "uncertain"
    return "likely_true"


def analyze_article(
    title,
    text,
    max_chunks=20,
    full_penalty_chunks=TITLE_MISMATCH_FULL_PENALTY_CHUNKS,
):
    chunks = split_into_chunks(text)
    if not chunks:
        return []

    similarities = sentence_title_similarity(title, chunks)
    citation_densities = sentence_citation_density(chunks)
    feature_counts = sentence_feature_counts(chunks)

    corroboration_max_sim = np.zeros(len(chunks), dtype=float)
    corroboration_support_counts = np.zeros(len(chunks), dtype=int)

    if CORROBORATION_INDEX is not None:
        (
            corroboration_max_sim,
            corroboration_support_counts,
        ) = chunk_corroboration_signals(chunks, CORROBORATION_INDEX)

    results = []
    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        sim = similarities[idx - 1]
        cit = citation_densities[idx - 1]
        features = {
            "citation_hits": feature_counts["citation_hits"][idx - 1],
            "evidence_hits": feature_counts["evidence_hits"][idx - 1],
            "hedge_hits": feature_counts["hedge_hits"][idx - 1],
            "absolute_hits": feature_counts["absolute_hits"][idx - 1],
            "absolute_in_quotes_hits": feature_counts[
                "absolute_in_quotes_hits"
            ][idx - 1],
            "absolute_outside_quotes_hits": feature_counts[
                "absolute_outside_quotes_hits"
            ][idx - 1],
            "attribution_hits": feature_counts["attribution_hits"][
                idx - 1
            ],
            "has_number": feature_counts["has_number"][idx - 1],
            "entity_mentions": feature_counts["entity_mentions"][idx - 1],
            "max_chunk_similarity": corroboration_max_sim[idx - 1],
            "support_articles": corroboration_support_counts[idx - 1],
        }

        score, reasons = score_sentence(
            sim,
            cit,
            features,
            chunk_position=idx,
            full_penalty_chunks=full_penalty_chunks,
        )
        label = label_from_score(score)

        results.append(
            {
                "chunk_id": idx,
                "similarity": sim,
                "citation_density": cit,
                "citation_hits": features["citation_hits"],
                "evidence_hits": features["evidence_hits"],
                "hedge_hits": features["hedge_hits"],
                "absolute_hits": features["absolute_hits"],
                "absolute_in_quotes_hits": features[
                    "absolute_in_quotes_hits"
                ],
                "absolute_outside_quotes_hits": features[
                    "absolute_outside_quotes_hits"
                ],
                "attribution_hits": features["attribution_hits"],
                "has_number": features["has_number"],
                "entity_mentions": features["entity_mentions"],
                "max_chunk_similarity": features["max_chunk_similarity"],
                "support_articles": features["support_articles"],
                "score": score,
                "label": label,
                "reasons": reasons,
                "chunk": chunk,
            }
        )

    return results


def print_results(title, article_label, rows):
    print("\nChunk-Level Analysis")
    print(f"Title: {title}")
    print(f"Article label (dataset): {article_label}")
    print(
        "\nChunk | Similarity | CitationDensity | CorrSim | Support | Score | HeuristicLabel"
    )
    print("-" * 94)

    for row in rows:
        print(
            f"{row['chunk_id']:>5} | "
            f"{row['similarity']:.4f}     | "
            f"{row['citation_density']:.4f}         | "
            f"{row['max_chunk_similarity']:.4f}  | "
            f"{row['support_articles']:^7} | "
            f"{row['score']:.1f}   | "
            f"{row['label']}"
        )
        print(
            "    counts: "
            f"abs_outside={row['absolute_outside_quotes_hits']}, "
            f"abs_in_quotes={row['absolute_in_quotes_hits']}, "
            f"attrib={row['attribution_hits']}, "
            f"entities={row['entity_mentions']}"
        )
        print(textwrap.fill(
            "reasons: " + ", ".join(row["reasons"]),
            width=100,
            initial_indent="    ",
            subsequent_indent="             ",
        ))
        wrapped = textwrap.fill(row["chunk"], width=100, initial_indent="    ", subsequent_indent="    ")
        print(wrapped)

    label_counts = {
        "likely_true": 0,
        "uncertain": 0,
        "likely_false": 0,
    }
    for row in rows:
        label_counts[row["label"]] += 1

    mean_score = float(np.mean([row["score"] for row in rows])) if rows else 0.0
    print("\nLabel Summary")
    print(
        "  likely_true={likely_true}, uncertain={uncertain}, likely_false={likely_false}".format(
            **label_counts
        )
    )
    print(f"  mean_chunk_score={mean_score:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title-full-penalty-chunks",
        type=int,
        default=TITLE_MISMATCH_FULL_PENALTY_CHUNKS,
        help="Number of leading chunks that keep full title mismatch penalty.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Row index in test set to analyze.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use raw test data here so punctuation remains available for sentence
    # splitting. Only drop rows missing title/text.
    test_data = load_test_data()

    sample_index = max(0, min(args.sample_index, len(test_data) - 1))
    sample = test_data.iloc[sample_index]

    global CORROBORATION_INDEX
    CORROBORATION_INDEX = build_chunk_corroboration_index(
        dataframe=test_data,
        sample_index=sample_index,
    )
    if CORROBORATION_INDEX is None:
        print("\nCorroboration index unavailable; skipping Rule 3.")
    else:
        print("\nCorroboration index ready (sampled article chunks).")

    print(
        "Title mismatch full-penalty chunks: "
        f"{args.title_full_penalty_chunks}"
    )

    title = sample["title"]
    text = sample["text"]
    article_label = sample["label"]

    rows = analyze_article(
        title,
        text,
        full_penalty_chunks=max(0, args.title_full_penalty_chunks),
    )

    output_path = Path(__file__).resolve().parent / "analysis_output.txt"
    with output_path.open("w", encoding="utf-8") as f:
        # Tee all print output to both terminal and file.
        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    s.write(data)
            def flush(self):
                for s in self._streams:
                    s.flush()

        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, f)
        try:
            print_results(title, article_label, rows)
        finally:
            sys.stdout = original_stdout

    print(f"\nOutput saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
