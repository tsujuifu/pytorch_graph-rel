## [**2021/12/20 Update**] Reimplementation by Author
Thanks to everyone's interest in this project and sorry for missing the original preprocessed data. <br>
It got lost in my previous lab, and I finally had time to reimplement it 😂. <br>
I also want to appreciate @LuoXukun for his [nice reply about reproducing](https://github.com/tsujuifu/pytorch_graph-rel/issues/6#issuecomment-615836064).

| [NYT](https://github.com/tsujuifu/pytorch_graph-rel/blob/master/_data/nyt.json) | Precision | Recall | F1 |
| :-: | :-: | :-: | :-: |
| GraphRel<sub>1p</sub> ([Paper](https://tsujuifu.github.io/pubs/acl19_graph-rel.pdf)) | 62.9 | 57.3 | 60.0 |
| GraphRel<sub>1p</sub> ([Reimplementation](https://github.com/tsujuifu/pytorch_graph-rel/blob/master/_snapshot/nyt_1p.pt)) | 60.9 | 59.2 | 60.1 |
| GraphRel<sub>2p</sub> ([Paper](https://tsujuifu.github.io/pubs/acl19_graph-rel.pdf)) | 63.9 | 60.0 | 61.9 |
| GraphRel<sub>2p</sub> ([Reimplementation](https://github.com/tsujuifu/pytorch_graph-rel/blob/master/_snapshot/nyt_2p.pt)) | 63.1 | 60.2 | 61.6 |


# [ACL'19 (Long)] GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction
A **PyTorch** implementation of GraphRel

[Paper](https://tsujuifu.github.io/pubs/acl19_graph-rel.pdf) | [Slide](https://tsujuifu.github.io/slides/acl19_graph-rel.pdf) | [Poster](https://github.com/tsujuifu/pytorch_graph-rel/raw/master/imgs/poster.png)

<img src='imgs/result.png' width='85%' />

## Overview
GraphRel is an implementation of <br> 
"[GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction](https://tsujuifu.github.io/pubs/acl19_graph-rel.pdf)" <br>
[Tsu-Jui Fu](http://tsujuifu.github.io/), [Peng-Hsuan Li](http://jacobvsdanniel.github.io/), and [Wei-Yun Ma](http://www.iis.sinica.edu.tw/pages/ma/) <br>
in Annual Meeting of the Association for Computational Linguistics (**ACL**) 2019 (Long)

<img src='imgs/overview.png' width='80%' />

In the 1st-phase, we **adopt bi-RNN and GCN to extract both sequential and regional dependency** word features. Given the word features, we **predict relations for each word pair** and the entities for all words. Then, in 2nd-phase, based on the predicted 1st-phase relations, we build complete relational graphs for each relation, to which we **apply GCN on each graph to integrate each relation’s information** and further consider the interaction between entities and relations.

## Requirements
This code is implemented under **Python3.8** and [PyTorch 1.7](https://pypi.org/project/torch/1.7.0). <br>
+ [tqdm](https://pypi.org/project/tqdm), [spaCy](https://spacy.io)

## Usage
```
python -m spacy download en_core_web_lg
python main.py --arch=2p
```
We also provide the [trained checkpoints](https://github.com/tsujuifu/pytorch_graph-rel/tree/master/_snapshot).

## Citation
```
@inproceedings{fu2019graph-rel, 
  author = {Tsu-Jui Fu and Peng-Hsuan Li and Wei-Yun Ma}, 
  title = {GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extractionn}, 
  booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL)}, 
  year = {2019} 
}
```

## Acknowledgement
+ [copy_re](https://github.com/xiangrongzeng/copy_re)
