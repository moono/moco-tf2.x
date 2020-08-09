# moco-tf2.x
* Unofficial reimplementation of [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning][1]
* Found many helpful implementations from [moco.tensorflow][3]
* Augmentation code are copied from [SimCLR - A Simple Framework for Contrastive Learning of Visual Representations][2]
* Trying to implement as much as possible in Tensorflow 2.x
  * Used MirroredStrategy with custom training loop ([tensorflow-tutorial][4])

## Note
* Difference between [official implementation][1]
  * 8 GPUs vs 4 GPUs
  * 53 Hours vs 130 hours (Unsupervised training time) - much slower than official one
* Batch normalization - tf
  * If one sets batch normalization layer as *un-trainable*, tf will normalize input with their moving mean & var, 
  even though you use `training=True`
* Lack of information about how to properly apply weight regularization within distributed environment

## Current result
* MoCo v1 - around 130 epoch results

|                | Plot                        |
| :------------: | :-------------------------: |
| InfoNCE        | ![InfoNCE][loss-graph]      |
| (K+1) Accuracy | ![Accuracy][accuracy-graph] |


[1]: https://github.com/facebookresearch/moco
[2]: https://github.com/google-research/simclr
[3]: https://github.com/ppwwyyxx/moco.tensorflow
[4]: https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_custom_training_loops
[loss-graph]: ./assets/moco-tf2.x-v1-loss.png
[accuracy-graph]: ./assets/moco-tf2.x-v1-accuracy.png
