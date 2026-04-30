from scipy.sparse import csr_matrix, hstack
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import (
    clean_normalize_dataset,
    read_test,
    read_train,
)


# Rule 1: title/body mismatch.
# If a title and article body score below this value,
# they are treated as a mismatch.
TITLE_MISMATCH_THRESHOLD = 0.30

# Rule 2: source/citation presence.
# If citation density is below this value,
# the article is treated as low-citation.
CITATION_THRESHOLD = 0.00075

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


def build_text_features(train_data, test_data):
    # Stick the title and body together into one block of text per article,
    # then score each word by how unique and important it is.
    # The model only learns word patterns from the training set,
    # not the test set.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    train_text = (train_data["title"] + " " + train_data["text"]).values
    test_text = (test_data["title"] + " " + test_data["text"]).values

    x_train_text = vectorizer.fit_transform(train_text)
    x_test_text = vectorizer.transform(test_text)
    return x_train_text, x_test_text


def compute_title_text_similarity(train_data, test_data):
    # A separate word scorer just for the title vs. body comparison.
    # Normalizing scores here makes similarity results more accurate.
    similarity_vectorizer = TfidfVectorizer(stop_words="english", norm="l2")

    # Only learn word patterns from the training articles.
    train_corpus = train_data["title"].tolist() + train_data["text"].tolist()
    similarity_vectorizer.fit(train_corpus)

    # Score each title and body separately so we can compare them.
    train_title_vec = similarity_vectorizer.transform(train_data["title"])
    train_text_vec = similarity_vectorizer.transform(train_data["text"])
    test_title_vec = similarity_vectorizer.transform(test_data["title"])
    test_text_vec = similarity_vectorizer.transform(test_data["text"])

    # Compute matching-row cosine only to avoid allocating an NxN matrix.
    train_similarity = rowwise_cosine_similarity(
        train_title_vec, train_text_vec
    )
    test_similarity = rowwise_cosine_similarity(test_title_vec, test_text_vec)
    return train_similarity, test_similarity


def rowwise_cosine_similarity(a_matrix, b_matrix):
    # Compare each title to its own article body, one pair at a time.
    # Doing it this way avoids trying to compare every title to every body
    # at once, which would run out of memory for a large dataset.
    numerator = a_matrix.multiply(b_matrix).sum(axis=1).A1
    a_norm = np.sqrt(a_matrix.multiply(a_matrix).sum(axis=1)).A1
    b_norm = np.sqrt(b_matrix.multiply(b_matrix).sum(axis=1)).A1
    denominator = a_norm * b_norm
    # If either the title or body is empty, the score defaults to 0.
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def build_title_mismatch_features(
    train_similarity,
    test_similarity,
    threshold=TITLE_MISMATCH_THRESHOLD,
):
    # Rule 1 feature output:
    # 1 means title/body mismatch, 0 means they are more aligned.
    train_mismatch = (train_similarity < threshold).astype(int)
    test_mismatch = (test_similarity < threshold).astype(int)

    # Bundle the similarity score and the mismatch flag together as two
    # extra columns that get added alongside the word-based features.
    train_extra = csr_matrix(
        list(zip(train_similarity, train_mismatch)),
        shape=(len(train_similarity), 2),
    )
    test_extra = csr_matrix(
        list(zip(test_similarity, test_mismatch)),
        shape=(len(test_similarity), 2),
    )
    return train_extra, test_extra


def compute_citation_density(dataframe):
    # Count how many source-style phrases appear in each article body,
    # then divide by text length so scores are comparable.
    phrase_counts = np.zeros(len(dataframe), dtype=float)
    text_series = dataframe["text"].fillna("")
    for phrase in SOURCE_PHRASES:
        phrase_counts += text_series.str.count(phrase).to_numpy(dtype=float)

    text_lengths = text_series.str.len().to_numpy(dtype=float)
    return np.divide(
        phrase_counts,
        np.maximum(text_lengths, 1.0),
        out=np.zeros_like(phrase_counts),
        where=True,
    )


def build_citation_features(
    train_data,
    test_data,
    threshold=CITATION_THRESHOLD,
):
    # Rule 2 feature output:
    # a low citation score means the article has fewer source-like phrases.
    train_citation_density = compute_citation_density(train_data)
    test_citation_density = compute_citation_density(test_data)

    train_missing_citation = (train_citation_density < threshold).astype(int)
    test_missing_citation = (test_citation_density < threshold).astype(int)

    train_citation_extra = csr_matrix(
        list(zip(train_citation_density, train_missing_citation)),
        shape=(len(train_citation_density), 2),
    )
    test_citation_extra = csr_matrix(
        list(zip(test_citation_density, test_missing_citation)),
        shape=(len(test_citation_density), 2),
    )
    return (
        train_citation_extra,
        test_citation_extra,
        train_citation_density,
        test_citation_density,
    )


def print_similarity_summary(
    name,
    similarity_scores,
    labels,
    threshold=TITLE_MISMATCH_THRESHOLD,
):
    # Rule 1 diagnostics.
    # This shows whether title/body similarity differs by class.
    mismatch_rate = (similarity_scores < threshold).mean()
    label_zero_mean = similarity_scores[labels == 0].mean()
    label_one_mean = similarity_scores[labels == 1].mean()

    print(f"\n{name} title-text similarity")
    print(f"  Mean similarity:        {similarity_scores.mean():.4f}")
    print(f"  Mismatch rate (< {threshold:.2f}): {mismatch_rate:.4f}")
    print(f"  Mean similarity label 0: {label_zero_mean:.4f}")
    print(f"  Mean similarity label 1: {label_one_mean:.4f}")


def evaluate_mismatch_rule(train_similarity, test_similarity, y_train, y_test):
    # Use training averages to decide which label behaves more like "false"
    # for this dataset. Lower similarity is treated as "more false-like".
    mean_label_zero = train_similarity[y_train == 0].mean()
    mean_label_one = train_similarity[y_train == 1].mean()
    false_label = 0 if mean_label_zero < mean_label_one else 1

    # Rule output: 1 means mismatch. Map mismatch to the inferred false label.
    mismatch_pred = (test_similarity < TITLE_MISMATCH_THRESHOLD).astype(int)
    y_rule_pred = np.where(mismatch_pred == 1, false_label, 1 - false_label)

    print("\nTitle Mismatch Rule Only")
    print(f"  Inferred false label: {false_label}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_rule_pred):.4f}")
    precision_false = precision_score(
        y_test,
        y_rule_pred,
        pos_label=false_label,
        zero_division=0,
    )
    recall_false = recall_score(
        y_test,
        y_rule_pred,
        pos_label=false_label,
        zero_division=0,
    )
    f1_false = f1_score(
        y_test,
        y_rule_pred,
        pos_label=false_label,
        zero_division=0,
    )
    print(
        f"  Precision (false label): {precision_false:.4f}"
    )
    print(f"  Recall (false label):    {recall_false:.4f}")
    print(f"  F1 (false label):        {f1_false:.4f}")
    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  {confusion_matrix(y_test, y_rule_pred, labels=[0, 1])}")


def print_citation_summary(
    name,
    citation_density,
    labels,
    threshold=CITATION_THRESHOLD,
):
    # Rule 2 diagnostics.
    # This shows whether citation/source density differs by class.
    low_citation_rate = (citation_density < threshold).mean()
    label_zero_mean = citation_density[labels == 0].mean()
    label_one_mean = citation_density[labels == 1].mean()

    print(f"\n{name} citation rule")
    print(f"  Mean citation density:      {citation_density.mean():.6f}")
    print(f"  Low-citation rate (< {threshold:.4f}): {low_citation_rate:.4f}")
    print(f"  Mean citation density label 0: {label_zero_mean:.6f}")
    print(f"  Mean citation density label 1: {label_one_mean:.6f}")


def main():
    # Load and clean the training and test articles.
    train_data = clean_normalize_dataset(read_train)
    test_data = clean_normalize_dataset(read_test)

    y_train = train_data["label"].values
    y_test = test_data["label"].values

    # Turn each article's words into numerical scores the models can use.
    x_train_text, x_test_text = build_text_features(train_data, test_data)

    # Score how closely each article's title matches its body (0 to 1).
    train_similarity, test_similarity = compute_title_text_similarity(
        train_data, test_data
    )

    # Add Rule 1 features (similarity score + mismatch flag)
    # alongside the word scores.
    train_extra, test_extra = build_title_mismatch_features(
        train_similarity, test_similarity
    )

    # Build Rule 2 features (citation density + low-citation flag).
    (
        train_citation_extra,
        test_citation_extra,
        train_citation_density,
        test_citation_density,
    ) = build_citation_features(train_data, test_data)

    x_train = hstack([x_train_text, train_extra, train_citation_extra])
    x_test = hstack([x_test_text, test_extra, test_citation_extra])

    # Print Rule 1 and Rule 2 diagnostics before model training.
    print_similarity_summary("Train", train_similarity, y_train)
    print_similarity_summary("Test", test_similarity, y_test)
    print_citation_summary("Train", train_citation_density, y_train)
    print_citation_summary("Test", test_citation_density, y_test)

    # Evaluate the mismatch rule by itself before model training.
    evaluate_mismatch_rule(train_similarity, test_similarity, y_train, y_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(max_iter=2000, random_state=1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=1
        ),
    }

    # Train each model and check how well it performs on unseen articles.
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        print(f"\n{model_name}")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(
            f"  Precision: "
            f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
        )
        print(
            f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}"
        )
        print(f"  F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
        print("\n  Confusion Matrix:")
        print(f"  {confusion_matrix(y_test, y_pred)}")


if __name__ == "__main__":
    main()
