# CLIPU2Net
Implementation of CLIPU^2Net for Referring Image Segmentation

[[`Website`](https://cjiang2.github.io/ref-uibvs/)][[`Paper`](https://arxiv.org/abs/2409.11518)] [[`Demo`](https://github.com/cjiang2/CLIPU2Net/blob/main/demo.ipynb)]


## Installation
To quickly deploy, clone the repository locally and install with

```
git clone https://github.com/cjiang2/CLIPU2Net.git
cd CLIPU2Net; pip install -r requirements.txt
```

The model requires the up-to-date `pytorch` to function. 

A [modified version of CLIP](https://github.com/cjiang2/CLIPU2Net/blob/main/clipu2net/models/clip/model.py#L241) is embedded in the repo to support multi-resolution image interpolation. 


## Acknowledgement
Thanks to the codebase from [CLIP](https://github.com/openai/CLIP), [CLIPSeg](https://github.com/timojl/clipseg).

Some image resources used in the demo here are from:
- [Spoon, Coffee Bean](https://pixabay.com/photos/coffee-bean-caffeine-coffee-4185338/)
