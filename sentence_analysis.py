import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import read_test


# Rule 1: how well each sentence matches the title.
SENTENCE_TITLE_MATCH_THRESHOLD = 0.14

# Rule 2: sentence citation density threshold.
SENTENCE_CITATION_THRESHOLD = 0.002

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
    }


def score_sentence(similarity, citation_density, features):
    # Higher score means more likely false.
    score = 0.0
    reasons = []

    if similarity < SENTENCE_TITLE_MATCH_THRESHOLD:
        score += 0.9
        reasons.append("low title match")

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

    if features["has_number"] == 1 and features["evidence_hits"] == 0:
        score += 0.4
        reasons.append("numeric claim without evidence words")

    if features["evidence_hits"] > 0:
        score -= 0.6
        reasons.append("contains evidence words")

    if features["citation_hits"] > 0:
        score -= 0.4
        reasons.append("contains citation phrase")

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


def analyze_article(title, text, max_sentences=20):
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    similarities = sentence_title_similarity(title, sentences)
    citation_densities = sentence_citation_density(sentences)
    feature_counts = sentence_feature_counts(sentences)

    results = []
    for idx, sentence in enumerate(sentences[:max_sentences], start=1):
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
        }

        score, reasons = score_sentence(sim, cit, features)
        label = label_from_score(score)

        results.append(
            {
                "sentence_id": idx,
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
                "score": score,
                "label": label,
                "reasons": reasons,
                "sentence": sentence,
            }
        )

    return results


def print_results(title, article_label, rows):
    print("\nSentence-Level Analysis")
    print(f"Title: {title}")
    print(f"Article label (dataset): {article_label}")
    print(
        "\nID | Similarity | CitationDensity | Score | HeuristicLabel"
    )
    print("-" * 66)

    for row in rows:
        print(
            f"{row['sentence_id']:>2} | "
            f"{row['similarity']:.4f}     | "
            f"{row['citation_density']:.4f}         | "
            f"{row['score']:.1f}   | "
            f"{row['label']}"
        )
        print(
            "    counts: "
            f"abs_outside={row['absolute_outside_quotes_hits']}, "
            f"abs_in_quotes={row['absolute_in_quotes_hits']}, "
            f"attrib={row['attribution_hits']}"
        )
        print(
            f"    reasons: {', '.join(row['reasons'])}"
        )
        print(f"    {row['sentence']}")


def main():
    # Use raw test data here so punctuation remains available for sentence
    # splitting. Only drop rows missing title/text.
    test_data = read_test.dropna(subset=["title", "text"]).reset_index(
        drop=True
    )

    sample_index = 0
    sample = test_data.iloc[sample_index]

    title = sample["title"]
    text = sample["text"]
    article_label = sample["label"]

    rows = analyze_article(title, text)
    print_results(title, article_label, rows)


if __name__ == "__main__":
    main()
