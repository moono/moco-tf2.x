# Implementation details

## [Author][Official Author]'s comments in [github][1] issue - Unsupervised training
* Training time
  * 53 hours per 200 epochs (8 GPUs)
  * around 6 min for first 1 epoch
* Loss
  * initial increase of loss graph
    * queue is being filled (initial noise is easier to discriminate)
    * task gets harder when queue is filled
* Dataset size
  * if dataset is smaller than Imagenet(1.28M), queue size must be adjusted, 
  because chances of containing positive samples in a queue will increase.
  * Positive samples should be removed from queue for the loss to make sense.
  * Another detailed question
    * Q
    > I have a small confusion regarding this issue (maybe I am missing something): can the situation occur when 
    the queue of negative examples contains some samples which are similar or exactly the same sample of the l_pos? 
    If it happens then should some labels of l_neg be one instead of zero?
    * A
    > Yes, it could happen. This only matters when the queue is too large. On ImageNet, the queue (65536) is ~6% 
    of the dataset size (1.28M), so the chance of having a positive in the queue is 6% at the first iteration of 
    each epoch, and reduces to 0% in the 6-th percentile of iterations of each epoch. This noise is negligible, 
    so for simplicity we don't handle it. If the queue is too large, or the dataset is too small, an extra 
    indicator should be used on these positive targets in the queue.
* [Linear learning rate scaling recipe][linear lr scaling recipe]
  * use in case of different environment setup(GPUs, batch sizes, ...)
* Batch size
  * 32 images per gpu is a legacy setting inherited from the original ResNet paper.
* Shuffle BN
  * Set encoder_q & encoder_k as training mode
    * ShuffleBN would be meaningless if the key encoder freezes BN
  * Q
  > Ditto. I kept wondering about SyncBN vs ShuffleBN as to whether the former can effectively prevent cheating. 
  SimCLR appears to be using SyncBN (referred to as "Global BN"). SyncBN is out of the box with PyTorch whereas 
  Shuffling BN requires a bit more hacking. The fact that Shuffling BN is chosen must mean that it is better? 
  (or that SyncBN wasn't ready at the time MoCo was designed?)
  * A
  > SyncBN is not sufficient in the case of MoCo: the keys in the queue are still from different batches, so the 
  BN stats in the current batch can serve as a signature to tell which subset the positive key may be in. Actually, 
  in a typical multi-GPU setting, ShuffleBN is faster than SyncBN. ShuffleBN only requires AllGather twice in the 
  entire network (actually, the input shuffle can be waived if implemented in data loader). SyncBN requires 
  communication per layer.
* Top-1 accuracy
  > This top-1 accuracy is the (K+1)-way (e.g., K=65536) classification accuracy of the dictionary look-up in 
  contrastive learning. It is not the ImageNet classification accuracy. Please finish the entire pipeline and 
  check what you get. (Btw, for 4-GPU training, we recommend --lr 0.015 --batch-size 128, though what you are 
  doing should be ok.)
* Measure convergence
  * Q
  > In the MoCo v2 paper, the models are far from convergence at 200ep (67.5% accuracy) and can be improved 
  by a lot when training 800ep (71.1%) or perhaps longer.
  * A
  > I suggest to monitor the truth performance of the feature by training a linear classifier or transferring 
  to a downstream task, because the loss/accuracy of the pretext task is not indicative 
  (an easier pretext task does not mean better features).

[1]: https://github.com/facebookresearch/moco
[linear lr scaling recipe]: https://arxiv.org/abs/1706.02677
[Official Author]: https://github.com/KaimingHe