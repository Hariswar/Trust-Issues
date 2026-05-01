# Model Explainability (Issue #4)

This is what I did for the explainability portion of the project. The main file is `explain.ipynb` in this folder.

## What I did

I used gradient saliency to figure out which words most influenced the BERT model's predictions. BERT represents each word as a vector of numbers internally. I ran the model on an article, then backpropagated from the predicted class logit through the embedding layer and measured the gradient at each word's position. Words with a larger gradient had more influence on that prediction.

I ran this across 200 test articles per dataset and averaged the scores by predicted class. That produced two ranked word lists: words that consistently mattered when the model said FAKE, and words that consistently mattered when it said REAL. Before ranking I filtered out stopwords (common words like "the", "and", "is") since they appear everywhere and don't tell you anything about real vs. fake patterns.

I also visualized BERT's self-attention weights for one sample article. Self-attention shows how much each token "pays attention" to every other token in the sequence. It's worth noting this is different from gradient saliency. Attention shows token relationships, not which words directly drove the prediction.

## What I tracked

All outputs saved to `outputs/figures/`.

| File | What it is |
|---|---|
| `confusion_matrices.png` | Confusion matrix for both datasets side by side |
| `per_class_metrics_dataset1.png` | Precision/recall/F1 bar chart for dataset 1 |
| `per_class_metrics_dataset2.png` | Same for dataset 2 |
| `top_words_dataset1.png` | Top 20 words driving FAKE and REAL predictions (dataset 1) |
| `top_words_dataset2.png` | Same for dataset 2 |
| `attention_heatmap.png` | BERT self-attention weights for a sample article |
| `accuracy_comparison.png` | BERT accuracy vs. Naive Bayes, SVM, Logistic Regression |
| `metrics_dataset1.json` | Full metrics: accuracy, precision, recall, F1, confusion matrix, top words |
| `metrics_dataset2.json` | Same for dataset 2 |

## Results

**Accuracy**

Dataset 1 hit 99% accuracy on 200 test samples (only 2 misclassifications). Dataset 2 hit 96% (8 misclassifications). Both models had strong precision and recall for both classes, which confirms the BERT training held up well.

Confusion matrices:
- Dataset 1: 101 real correctly classified, 97 fake correctly classified, 2 fake articles called real, 0 real articles called fake
- Dataset 2: 112 real correctly classified, 80 fake correctly classified, 7 real articles called fake, 1 fake article called real

**Word importance patterns**

Dataset 2 had the clearest word patterns. The top FAKE words included "shocking", "controversial", "alleged", "billionaire", and "fake" itself, which are all classic sensationalist clickbait language. The top REAL words included "congressional", "candidate", "reportedly", and "cnn", which read like neutral journalistic vocabulary. That separation is exactly what you'd want to see.

Dataset 1's top FAKE words were led by "nikki" and "haley", which is most likely a sampling artifact. The 200 articles for that run happened to include a cluster of Nikki Haley fake news stories, so her name scored high. It's not wrong, just specific to that sample rather than a general pattern the model learned.

"Dem" showed up as a top REAL word in both datasets, which is interesting. It shows up a lot in news headlines as shorthand (e.g. "Dem senator") and the model seems to have picked up on that style being associated with real reporting.

## LLM explanations

For the plain-English explanation piece I used qwen2.5:32b running locally through ollama. For each sample article I passed in the article text, what BERT predicted, and the top influential words with their gradient scores. The model then wrote 2-3 sentences explaining what linguistic patterns or credibility signals BERT likely noticed that led to that classification.

I picked qwen2.5:32b because I had 31 GB of RAM and the model is about 20 GB, so it fit without issues. All explanations are saved in the notebook output at the bottom of `explain.ipynb` and can be read there without running anything.

I ran this on 5 articles per dataset. A few notable examples:

**Missouri Republican article (FAKE, predicted correctly).** The top words were "idiot", "horrific", and "contentious". The LLM explanation correctly identified that the article leaned heavily on emotionally charged, informal language rather than neutral reporting. This was one of the clearest cases where the word importance lined up with an obvious stylistic reason.

**Schiff article (REAL, predicted FAKE).** This was the one misclassification in the dataset 2 LLM sample. The title was "schiff threatens to violate constitution twice with independent counse..." which genuinely sounds like a fake news headline. The model got tripped up by the sensationalist phrasing even though the underlying article was real. The LLM explanation picked up on this correctly, noting that words like "threatens" and "violate" made the article look fabricated. It's a good example of where the model's limitation shows since it's pattern-matching on writing style, and some real articles use provocative language.

**Diversity visa / Reuters article (FAKE, predicted correctly).** This one was interesting because "reuters" showed up as one of the influential words. The article was fake but cited Reuters in the text, and the model still called it fake. That suggests it was picking up on other patterns in the article rather than the source citation alone.

## Files

```
explain.ipynb          the notebook
outputs/figures/       all generated plots and JSON
EXPLAINABILITY.md      this file
```
