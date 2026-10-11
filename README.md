# RQR: Reasoning-Quality Regularization for Multimodal Fake News Detection

This repository accompanies a manuscript that is currently under review.

## Release Note

Only the core implementation is provided at this stage. To protect the integrity of the
ongoing review, the following are withheld and will be released upon acceptance:

- benchmark datasets and their preprocessing
- configuration files and runtime settings
- trained model checkpoints
- the scripts that drive visual fact extraction, reasoning generation and detector training

The code published here is therefore not intended to be run end to end in its current form.

## Overview

RQR structures the reasoning of a multimodal large language model and uses the estimated
quality of that reasoning to reweight detector training. The method has three parts.

1. **Structured prompting.** A multimodal model reads the image under a constrained JSON
   schema and returns a quadruple `K(I, T) = <E, U, B, N>`, namely entities, unary facts,
   binary facts, and negated facts that the news text asserts but the image does not support.
2. **Multi-step veracity judgment.** A language model receives the news text together with the
   extracted facts and reasons along an explicit path, returning a verdict and a list of
   supporting reasons.
3. **Quality-weighted detector.** A text encoder reads the news text concatenated with its
   reasons. A quality network scores the reasoning, and that score weights a per-sample
   contrastive loss, so better-reasoned samples contribute more to the learned representation.

## Repository Layout

| Path | Description |
|---|---|
| `RQR_run/cot_prompt_list.py` | prompting templates for both generation stages |
| `RQR_run/detector_func.py` | detector definition, including the quality network and the quality-weighted contrastive loss |
| `RQR_run/resp_process.py` | schema of the extracted facts and post-processing of model responses |
