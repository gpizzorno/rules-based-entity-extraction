# A rules-based approach to entity extraction from Latin texts

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The code in this repository is designed to extract structured information (entities and their relationships) from Medieval Latin texts, leveraging both linguistic rules (via chunking) and machine learning. It is particularly tailored for historical or philological research, where both precision (via rules) and adaptability (via ML) are valuable. The two main Jupyter notebooks, `chunk_text.ipynb` and `train_model.ipynb`, form the core of the workflow.

## 1. Rule-Based Chunking and Preprocessing

Implemented in `chunk_text.ipynb` this step preprocesses the raw Latin text, applying rule-based chunking, and generating HTML-based visualizations to verify the results.

Corpus data is loaded from `data/lines_clean_dev.txt`. The chunking relies on a sample of manually-annotated tags including a gloss dictionary, noun, and proper noun lists, and measurement units from the `data` directory. The workflow uses the Stanza NLP library to tokenize, lemmatize, and POS-tag the corpus. It applies custom overrides to POS tags and features based on the loaded dictionaries and some hardcoded rules (e.g., marking certain words as objects, units, or locations).

Pre-defined chunking rules are loaded from `chunk_rules.chk` into an NLTK RegexpParser grammar and each sentence is parsed into a chunk tree according to them. The rules are quite involved and hard to read, so I developed a [Tree-sitter parser for NLTK chunk syntax](https://github.com/gpizzorno/tree-sitter-chunk-grammar) and a [syntax highlighting module for Atom](https://github.com/gpizzorno/atom-language-chunkgrammar) to help.

Custom visualization functions convert the chunked sentences into HTML, mapping chunk labels to colors and descriptions (see the key and colours dictionaries). The HTML is saved as `chunks.html` to the `results-viz` directory for inspection. A shallow chunking step simplifies the trees to only keep top-level (level 1) tags, mapping detailed chunk labels to broader categories (e.g., `OP` to `OBJECT`). The shallow trees are saved to `parsed_sentences_shallow.pickle` and visualized in `chunks_shallow.html`.

### Visualization example:

<img src="data/viz-example.svg">


## 2. Machine Learning-Based Chunking
In `train_model.ipynb`, we take the output of the rule-based chunker and train a machine learning model to perform chunking/entity extraction.

The shallow chunked trees produced by `chunk_text.ipynb` are converted into IOB ([Inside-Outside-Beginning](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging))) format. The results are saved to `results/chunks_iob.txt`.

A rich `Feature Extraction` function is defined that, for each token, considers:

- The word, lemma, POS tags, and grammatical features.
- Contextual information from neighboring tokens (previous/next words, tags, lemmata).
- Custom features such as capitalization, object/unit/location markers, etc.

The workflow implements a custom NLTK chunker class (`NamedEntityChunker`) that uses a [ClassifierBasedTagger](https://www.nltk.org/api/nltk.tag.sequential.html#nltk.tag.sequential.ClassifierBasedTagger) trained on the extracted features.

The model is then trained, persisted to disk as `models/chunking_model.pickle`, and its performance is evaluated:

```mermaid
xychart-beta
    title "Training Results"
    x-axis "Runs" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    y-axis "Score" 0.0 --> 1.0
    line "Accuracy" [0.5617816091954023, 0.6556836902800659, 0.6763888888888889, 0.637434554973822, 0.6943661971830986, 0.6458923512747875, 0.608, 0.66, 0.6613138686131387, 0.6210235131396957, 0.6848184818481848, 0.6262773722627737, 0.6875800256081946, 0.5996955859969558, 0.6454413892908828]
    line "Precision" [0.11827956989247312, 0.13043478260869565, 0.17204301075268819, 0.14893617021276595, 0.13978494623655913, 0.10869565217391304, 0.11458333333333333, 0.1348314606741573, 0.11956521739130435, 0.14130434782608695, 0.18072289156626506, 0.15625, 0.1702127659574468, 0.09900990099009901, 0.12121212121212122]
    line "Recall" [0.6875, 0.8, 0.8, 0.8235294117647058, 0.7222222222222222, 0.625, 0.8461538461538461, 0.8, 0.6470588235294118, 0.7647058823529411, 0.75, 1.0, 0.9411764705882353, 0.9090909090909091, 0.8571428571428571]
    line "F-measure" [0.2018348623853211, 0.22429906542056072, 0.28318584070796465, 0.25225225225225223, 0.23423423423423423, 0.18518518518518517, 0.20183486238532108, 0.23076923076923073, 0.2018348623853211, 0.2385321100917431, 0.29126213592233013, 0.27027027027027023, 0.2882882882882883, 0.17857142857142858, 0.21238938053097348]
```
Key: #3498db = Accuracy, #2ecc71 = Precision, #e74c3c = Recall, #f1c40f = F-measure


## 3. Supporting Files and Directories

- `data`: Contains dictionaries and lists (glosses, nouns, proper nouns, measurement units) used for tagging and feature extraction.
- `rules`: Contains the rule-based chunking grammars.
- `results`: Stores intermediate and final outputs (lemmata, parsed sentences, IOB tags).
- `results-viz`: Contains HTML visualizations and CSS for inspecting chunking results.
- `models`: Stores the trained machine learning chunker.

## License
The project is licensed under the MIT License (LICENSE), allowing free use, modification, and distribution. See the LICENSE file in the top distribution directory for the full license text.













