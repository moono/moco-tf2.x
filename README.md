# moco-tf2.x
* Unofficial reimplementation of [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning][1]
* Found many helpful implementations from [moco.tensorflow][3]
* Augmentation code are copied from [SimCLR - A Simple Framework for Contrastive Learning of Visual Representations][2]
* Trying to implement as much as possible in Tensorflow 2.x
  * Used MirroredStrategy with custom training loop ([tensorflow-tutorial][4])

## Note
* Difference between [official implementation][1]
  * 8 GPUs vs 4 GPUs
  * 53 Hours vs 147 hours (Unsupervised training time) - much slower than official one
* Batch normalization - tf
  * If one sets batch normalization layer as *un-trainable*, tf will normalize input with their moving mean & var, 
  even though you use `training=True`
* Lack of information about how to properly apply weight regularization within distributed environment

## Result (Result _lower_ than [official one][1])
* MoCo v1
  * Could not reproduce same accuracy (Linear classification protocol on Imagenet) result as official one.

|         | InfoNCE                      | (K+1) Accuracy               |
| :-----: | :--------------------------: | :--------------------------: |
| MoCo V1 | ![InfoNCE][moco-v1-info-nce] | ![Accuracy][moco-v1-k+1-acc] |

|         | Train Accuracy                       | Validation Accuracy                 |
| :-----: | :----------------------------------: | :---------------------------------: |
| lincls | ![Accuracy][moco-v1-lincls-train-acc] | ![Accuracy][moco-v1-lincls-val-acc] |

* Comparison with official result

| ResNet-50             | pre-train <br>epochs | pre-train <br>time | MoCo v1 <br>top-1 acc.|
| :-------------------: | :------------------: | :----------------: | :-------------------: |
| Official <br>Result   | 200                  | 53 hours	        | 60.6                  |
| This repo <br>Result  | 200                  | 147 hours	        | 50.8                  |


[1]: https://github.com/facebookresearch/moco
[2]: https://github.com/google-research/simclr
[3]: https://github.com/ppwwyyxx/moco.tensorflow
[4]: https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_custom_training_loops
[moco-v1-info-nce]: ./assets/moco-tf2.x-v1-loss.png
[moco-v1-k+1-acc]: ./assets/moco-tf2.x-v1-accuracy.png
[moco-v1-lincls-train-acc]: ./assets/moco-tf2.x-v1-lincls-train-accuracy.png
[moco-v1-lincls-val-acc]: ./assets/moco-tf2.x-v1-lincls-val-accuracy.png
