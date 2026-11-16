# Third-party components and attribution

The project integrates external libraries, pretrained models, and datasets. Their own licenses, model cards, and data access terms apply independently of this repository.

| Component | Role | Upstream |
| --- | --- | --- |
| PyTorch / torchvision | Neural model training and inference | https://github.com/pytorch/pytorch |
| RAD-DINO-MAIRA-2 | Chest radiograph image encoder | https://huggingface.co/microsoft/rad-dino-maira-2 |
| Transformers | Pretrained image and text model loading | https://github.com/huggingface/transformers |
| FAISS | Vector similarity search | https://github.com/facebookresearch/faiss |
| scikit-learn | Learned gate and classifier refinement tools | https://github.com/scikit-learn/scikit-learn |
| Gradio | Local research interface | https://github.com/gradio-app/gradio |
| CheXbert | Clinical report label agreement evaluation | https://github.com/stanfordmlgroup/CheXbert |
| RadGraph | Clinical report entity/relation evaluation | https://github.com/Stanford-AIMI/radgraph |

IU-Xray/Indiana University chest radiographs and source reports, CheXpert-related training data, trained checkpoints, and generated study reports are not bundled in this source release. Obtain any required access and permissions from their original providers.

The local CheXbert checkout used while preparing this publication was at commit `6d22a96d73f18d0a7cf5b0dbebdac50cf8e4c1aa`. The bootstrap requirements document the RadGraph revision `87f11a1ff4d2046a838be5f0243857d93780ddec`. These checkouts and their weights are excluded from Git; inspect the evaluator scripts and `requirements-radgraph-bootstrap.txt` before installing them separately.

No external model, dataset, or vendored dependency is relicensed by publishing this repository. Training notebooks describe research workflows and do not grant access to the underlying datasets.
