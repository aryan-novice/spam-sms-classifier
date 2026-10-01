"""Builds and runs notebooks/02_improved_classifier.ipynb  (python notebooks/build_notebook.py)"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Spam SMS classifier, version 2

My first version (`01_tutorial_baseline.ipynb`) followed a tutorial: TF-IDF + logistic regression, 96.6% test accuracy.
This notebook asks: **is 96.6% actually good?** It turns out the answer is no, and fixing that taught me more than building the first model."""),
    code("""import sys, json
import pandas as pd
sys.path.insert(0, "../src")
df = pd.read_csv("../data/spam.csv")
print(df.shape)
df["Category"].value_counts(normalize=True).round(3)"""),
    md("""## Problem 1: accuracy hides the class imbalance
Only about 13% of messages are spam. A "model" that says **ham every time** is already about 87% accurate.
So 96.6% accuracy is not as impressive as it sounds. From here on, **spam is the positive class**, and I track:
- **Recall:** what share of spam did we catch?
- **Precision:** when we flag something as spam, how often are we right?"""),
    md("""## Problem 2: duplicate messages leak into the test set
The same message (e.g. a common spam blast) can appear many times. If one copy lands in training and another in testing,
the model is tested on text it has already seen."""),
    code("""dupes = df.duplicated(subset="Message").sum()
print(f"{dupes} duplicate messages ({dupes / len(df):.1%} of the data)")
df[df.duplicated(subset="Message", keep=False)].groupby("Message").size().sort_values(ascending=False).head(5)"""),
    md("""## Fixes
1. Remove duplicates **before** splitting.
2. Stratified train/test split, so both sets keep the same spam ratio.
3. Compare models with **5-fold cross-validation** on the training set, then check once on the untouched test set.
4. Try the models from my "future improvements" list: Naive Bayes, a class-balanced logistic regression, and a linear SVM on word + character n-grams.

All of this lives in `src/train.py`:"""),
    code("""!cd .. && python src/train.py 2>/dev/null | head -30"""),
    code("""m = json.load(open("../reports/metrics.json"))
pd.DataFrame(m["test_set"]).T.mul(100).round(1)"""),
    md("""**The original model let about a third of spam through** (recall around 68%), even with 95.7% accuracy on the
de-duplicated test set. The linear SVM on words + character n-grams catches about 92% of spam. Character n-grams help with
spammer tricks like `FR33`, `£1000` and `www.` links.

![](../reports/figures/model_comparison.png)"""),
    md("""## Problem 3: which mistake is worse?
For a spam filter, hiding a **real** message (a false positive) is worse than letting one spam through.
So instead of the default 0.5 cut-off, I pick the threshold that keeps **precision ≥ 99%** with as much recall as possible."""),
    code("""pd.DataFrame({"threshold 0.5": m["threshold"]["at_0.5"], f"threshold {m['threshold']['value']}": m["threshold"]["tuned"]}).mul(100).round(1)"""),
    code("""m["confusion_matrix_tuned"]"""),
    md("![](../reports/figures/precision_recall.png)\n\n![](../reports/figures/confusion_matrix.png)"),
    md("## What does it still get wrong?"),
    code("""pd.set_option("display.max_colwidth", 120)
pd.read_csv("../reports/error_samples.csv").head(10)"""),
    md("""## Try it, including a modern Indian scam"""),
    code("""!cd .. && python src/predict.py "Congratulations! You won a free iPhone, call 09061701461 now" "are we still meeting at 6?" "URGENT: your account will be blocked, update KYC at bit.ly/xyz" "Your OTP for login is 482910. Do not share it." """),
    md("""**Limitation, found by testing:** the KYC scam gets a low spam score. This dataset is from 2011 and mostly UK messages,
so it has never seen today's Indian scams (KYC updates, fake electricity bills, UPI collect requests). That's **data drift**:
a model is only as current as its training data. A next step would be to collect and label recent Indian scam SMS.

## Summary
| | v1 (tutorial) | v2 (this notebook) |
|---|---|---|
| Duplicates | in both train and test | removed before splitting |
| Main metric | accuracy | spam precision and recall |
| Model choice | one model | 4 models, 5-fold cross-validation |
| Decision threshold | 0.5 | tuned for ≥ 99% precision |
| Spam caught | about 68% | about 89%, with 99% precision |"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "02_improved_classifier.ipynb")
print("Saved notebooks/02_improved_classifier.ipynb")
