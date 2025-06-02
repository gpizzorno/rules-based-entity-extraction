# A rules-based approach to entity extraction from Latin texts

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The code in this repository is designed to extract structured information (entities and their relationships) from Medieval Latin texts, leveraging both linguistic rules (via chunking) and machine learning. It is particularly tailored for historical or philological research, where both precision (via rules) and adaptability (via ML) are valuable. The two main Jupyter notebooks, `chunk_text.ipynb` and `train_model.ipynb`, form the core of the workflow.

## 1. Rule-Based Chunking and Preprocessing

Implemented in `chunk_text.ipynb` this step preprocesses the raw Latin text, applying rule-based chunking, and generating HTML-based visualizations to verify the results.

Corpus data is loaded from `data/lines_clean_dev.txt`. The chunking relies on a sample of manually-annotated tags including a gloss dictionary, noun, and proper noun lists, and measurement units from the `data` directory. The workflow uses the Stanza NLP library to tokenize, lemmatize, and POS-tag the corpus. It applies custom overrides to POS tags and features based on the loaded dictionaries and some hardcoded rules (e.g., marking certain words as objects, units, or locations).

Pre-defined chunking rules are loaded from `chunk_rules.chk` into an NLTK RegexpParser grammar and each sentence is parsed into a chunk tree according to them. The rules are quite involved and hard to read, so I developed a [Tree-sitter parser for NLTK chunk syntax](https://github.com/gpizzorno/tree-sitter-chunk-grammar) and a [syntax highlighting module for Atom](https://github.com/gpizzorno/atom-language-chunkgrammar) to help.

Custom visualization functions convert the chunked sentences into HTML, mapping chunk labels to colors and descriptions (see the key and colours dictionaries). The HTML is saved as `chunks.html` to the `results-viz` directory for inspection. A shallow chunking step simplifies the trees to only keep top-level (level 1) tags, mapping detailed chunk labels to broader categories (e.g., `OP` to `OBJECT`). The shallow trees are saved to `parsed_sentences_shallow.pickle` and visualized in `chunks_shallow.html`.

### Visualization example:

<style>
.sample-block {
	font-family: "Menlo", "Helvetica Neue", Helvetica, Arial, sans-serif;
	font-size: 15px;
	line-height: 1.5;
	color: #a7a7a7;
	background: #222;
}
.sentence {
  display: flex;
  align-items: stretch;
  margin-bottom: 20px;
  border-bottom: 1px dashed #3a3a3a;
}
.container {
    display: inline-flex;
  align-items: flex-end;
}
.token {
	display: inline-flex;
  margin: 0 3px;
}
.unit {
  display: inline-flex;
	flex-direction: column;
}
.unit_label {
  font-size: 10px;
  text-align: center;
  line-height: 1;
  padding: 2px 5px;
  display: flex;
  margin-left: auto;
  margin-right: auto;
  background-color: #222;
  margin-bottom: -5px;
  z-index: 2;
}
.hi {
	background: #564900;
}
.unit_content {
  display: flex;
  align-items: flex-end;
  border-radius: 3px;
  border-top: 1px solid #848484;
  padding-top: 2px;
  min-width: 20px;
  justify-content: center;
  border-right: 1px solid #848484;
  border-left: 1px solid #848484;
  padding-left: 1px;
  padding-right: 1px;
  margin-right: 1px;
  margin-left: 1px;
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}
.line_no {
  display: inline-flex;
  font-size: 12px;
  font-weight: 900;
  align-content: center;
  justify-content: center;
  align-items: flex-end;
  background-color: #383737;
  color: #656464;
  width: 30px;
  margin-right: 10px;
  padding-bottom: 2px;
  flex-shrink: 0;
}

.green {
	color: rgb(24, 222, 24) !important;
}

.red {
	color: rgb(235, 13, 13) !important;
}

.chart {
    padding: 20px 30px 20px 20px;
    background-color: #333333;
}

</style>
<div class="sample-block">
    <div class="sentence">
        <div class="line_no red">1</div>
        <div class="container">
            <div class="token" title="CCONJ-Feats=0">Et</div>
            <div class="unit">
                <div class="unit_label" style="color: None;" title="Particle - list marker"> pList</div>
                <div class="unit_content " style="border-color: None;">
                    <div class="token" title="ADV-Degree=Abs|Lemma=primo">primo</div>
                </div>
            </div>
            <div class="unit">
                <div class="unit_label" style="color: None;" title="Particle - context"> pCtx</div>
                <div class="unit_content " style="border-color: None;">
                    <div class="token"
                        title="AUX-Aspect=Perf|InflClass=LatAnom|Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin|Lemma=sum">
                        fuerunt</div>
                    <div class="token"
                        title="VERB-Aspect=Perf|Case=Nom|Gender=Neut|InflClass=LatX|InflClass[nominal]=IndEurO|Number=Plur|VerbForm=Part|Voice=Pass|Lemma=invenio">
                        inventa</div>
                </div>
            </div>
            <div class="unit">
                <div class="unit_label" style="color: None;" title="Particle - locative"> pLoc</div>
                <div class="unit_content " style="border-color: None;">
                    <div class="token" title="ADP-Feats=0|Lemma=in">in</div>
                </div>
            </div>
            <div class="unit">
                <div class="unit_label" style="color: #a27230;" title="Location - room"> LOC-rm</div>
                <div class="unit_content " style="border-color: #a27230;">
                    <div class="token"
                        title="NOUN-Case=Abl|Gender=Fem|InflClass=IndEurA|Number=Sing|Object=1|Function=Location|Type=room">
                        coquina</div>
                </div>
            </div>
            <div class="unit">
                <div class="unit_label" style="color: #a27230;" title="Location - landmark"> LOC-lm</div>
                <div class="unit_content " style="border-color: #a27230;">
                    <div class="token"
                        title="NOUN-Case=Gen|Gender=Fem|InflClass=IndEurU|Number=Sing|Object=1|Function=Location|Type=lm">
                        domus</div>
                </div>
            </div>
            <div class="unit">
                <div class="unit_label" style="color: None;" title="Particle - reference"> pRef</div>
                <div class="unit_content " style="border-color: None;">
                    <div class="token"
                        title="VERB-Aspect=Perf|Case=Gen|Gender=Masc|InflClass=LatX|InflClass[nominal]=IndEurO|Number=Sing|VerbForm=Part|Voice=Pass|Lemma=dico">
                        dicti</div>
                </div>
            </div>
            <div class="token" title="NOUN-Case=Gen|Gender=Masc|InflClass=IndEurO|Number=Sing">Guillelmi</div>
            <div class="unit">
                <div class="unit_label" style="color: #2ebdad;" title="Attribute - adjectival"> ATT-Adj</div>
                <div class="unit_content " style="border-color: #2ebdad;">
                    <div class="token" title="ADJ-Case=Nom|Gender=Neut|InflClass=IndEurO|Number=Plur">hec</div>
                </div>
            </div>
            <div class="token" title="NOUN-Case=Nom|Gender=Neut|InflClass=IndEurO|Number=Plur|Lemma=bonum">bona</div>
        </div>
    </div>
    <div class="sentence">
        <div class="line_no red">3</div>
        <div class="container">
            <div class="token" title="ADV-Feats=0">Item</div>
            <div class="unit">
                <div class="unit_label" style="color: #3aea74;" title="Object phrase"> OP</div>
                <div class="unit_content " style="border-color: #3aea74;">
                    <div class="unit">
                        <div class="unit_label" style="color: #006cff;" title="Quantity"> QT</div>
                        <div class="unit_content " style="border-color: #006cff;">
                            <div class="token"
                                title="NUM-Case=Nom|Gender=Neut|InflClass=LatPron|NumType=Card|NumValue=1|Number=Sing|PronType=Ind">
                                unum</div>
                        </div>
                    </div>
                    <div class="unit">
                        <div class="unit_label" style="color: #b2d29f;" title="Object"> OBJ</div>
                        <div class="unit_content " style="border-color: #b2d29f;">
                            <div class="token" title="NOUN-Case=Nom|Gender=Neut|InflClass=IndEurO|Number=Sing|Object=1">
                                scudellium</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>


## 2. Machine Learning-Based Chunking
In `train_model.ipynb`, we take the output of the rule-based chunker and train a machine learning model to perform chunking/entity extraction.

The shallow chunked trees produced by `chunk_text.ipynb` are converted into IOB ([Inside-Outside-Beginning](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging))) format. The results are saved to `results/chunks_iob.txt`.

A rich `Feature Extraction` function is defined that, for each token, considers:

- The word, lemma, POS tags, and grammatical features.
- Contextual information from neighboring tokens (previous/next words, tags, lemmata).
- Custom features such as capitalization, object/unit/location markers, etc.

The workflow implements a custom NLTK chunker class (`NamedEntityChunker`) that uses a [ClassifierBasedTagger](https://www.nltk.org/api/nltk.tag.sequential.html#nltk.tag.sequential.ClassifierBasedTagger) trained on the extracted features.

The model is then trained, persisted to disk as `models/chunking_model.pickle`, and its performance is evaluated:

<img class="chart" src="https://mermaid.ink/img/pako:eNptVUtv2zAM_iuGzk4gPiSKOQwYNuw2YOh22tqDl3ipscQZHAdoV_S_j5bSJi6myJL9hS99JO0ntz5sWrdyt31l4-Fxfd8M4-JnOzYFGbtx11a37tvQdH3Xb6ub9njajcfbF41F89AdTeDm1BtY_YC6wrqiuuK6CnUV60rqKtWV1hV4u-x_MAEwCeC7YuTxxcjX9WFozYpf-mqxeFfB0heJXddPUbxfr09Ds36cHPlliCAJolfQwB7Nol_GEGKiqB6T9zFoxiRSuoyCkTBxCKxCCTFDyhQjqEAirylmjENSpAAoLElCxnzKWxGIQEApprJLxhD8pGOARg0FS5zg6ipyEUVI8LwXOQkWOQbzAsrZR9DJTEgv-zkwZnOoqD4lTHcznr4M7bo7doe-EAWQUDRENXEWgnxcIG8MSELzZGZjPhsIGurBS8CYLIIMclIyrj0aDcZpkSSVxBYhknGukNmHsykEIZ0cZBCMRLoaRZ0TAUcfhSFIwWCmS8UPvwk0g8kba0khRCMx-EyTPeD5EK-RCnPMfHlVf31BFsT5D-cs3rTrZrcrFMZz8tNlsRSjMoBEFl-yYkHNRslyUeQIgS7rq52i_Wqt2HkDZszMWD9Mt1deJ5FMnvrZr9gPAoyXdX68T4t92xxPueHshOit7jlZPlMgLD4RGY2sGBi9VaTkA6ElLoXEXrxo5FI4GHA2c0xIPJ8ld2E2c9m_cV46DMlL1MtK_5EsYZ7vLbNgbV0wBWsroKAWjFV0UbbKuJ4FnPrnMkuQMueuxAPmSCn5YO8HsSDuXO22Q7dxq3E4tbXbt8O-mR7d08T0rRvv273Ru7LbTTP8nl6az6bzp-m_Hw77F7XhcNreu9WvZne0p9OfTTO2H7tmOzT7V3Ro-007fDic-tGt7JWXjbjVk3twqwXHpViXoUUVST1RrN2jW4EVTSINlkRre-Egz7X7mx1bIZGI2kmsyz0rYe3aTTcehs_le5A_C8__AOznYWE" width=650>



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
## 3. Supporting Files and Directories

- `data`: Contains dictionaries and lists (glosses, nouns, proper nouns, measurement units) used for tagging and feature extraction.
- `rules`: Contains the rule-based chunking grammars.
- `results`: Stores intermediate and final outputs (lemmata, parsed sentences, IOB tags).
- `results-viz`: Contains HTML visualizations and CSS for inspecting chunking results.
- `models`: Stores the trained machine learning chunker.

## License
The project is licensed under the MIT License (LICENSE), allowing free use, modification, and distribution. See the LICENSE file in the top distribution directory for the full license text.













