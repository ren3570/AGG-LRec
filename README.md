# AGG-LRec

**Anchor Node Guided Global–Local Graph Neural Networks for Multimedia Recommendation**

[![Paper](https://img.shields.io/badge/Paper-JVCIR%202026-blue)](https://doi.org/10.1016/j.jvcir.2026.104835)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/ren3570/AGG-LRec)

Official implementation of the paper:

> **Anchor Node Guided Global–Local Graph Neural Networks for Multimedia Recommendation (AGG-LRec)**
> Journal of Visual Communication and Image Representation (JVCIR), 2026.

---

## Overview

Graph Neural Networks (GNNs) have achieved remarkable success in multimedia recommendation by exploiting high-order collaborative signals from user–item interaction graphs. Existing studies mainly focus on sparse interaction scenarios, while the challenges caused by dense interactions remain underexplored.

We argue that dense interactions introduce two critical issues:

* **Bottleneck Problem:** excessive neighborhood aggregation weakens the ability to capture personalized preferences.
* **Noise Propagation:** dense connections increase the likelihood of propagating irrelevant information.

To address these issues, we propose **AGG-LRec**, a novel recommendation framework that jointly models global and local collaborative information through anchor nodes.

---

## Training

```bash
python main.py
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{Ren2026AnchorNG,
  title={Anchor node guided global-local graph neural networks for multimedia recommendation},
  journal={Journal of Visual Communication and Image Representation},
  year={2026},
  doi={10.1016/j.jvcir.2026.104835},
  url={https://api.semanticscholar.org/CorpusID:288565834}
}
```
