import numpy as np
import tensorflow as tf


class StepDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, lr_decay_rate, n_images, epochs, boundaries_in_epoch, global_batch_size, name=None):
        super(StepDecay, self).__init__()
        # compute boundary edges
        max_step_size = int(np.ceil(n_images / global_batch_size))
        steps_per_epoch = int(np.ceil(max_step_size / epochs))
        boundaries_in_step = [0] + [v * steps_per_epoch for v in boundaries_in_epoch] + [max_step_size]
        n_boundaries = len(boundaries_in_step)
        learning_rates = [initial_lr] + [initial_lr * (lr_decay_rate ** (p + 1)) for p in range(n_boundaries - 2)]

        # assign
        self.boundaries_in_step = tf.convert_to_tensor(boundaries_in_step)
        self.learning_rates = tf.convert_to_tensor(learning_rates)
        self.n_boundaries = n_boundaries
        self.name = name

    def __call__(self, step):
        global_step_recomp = tf.cast(step, dtype=tf.int32)

        pos = tf.zeros([], dtype=tf.int32)
        for ii in range(self.n_boundaries - 1):
            boundary_left = tf.convert_to_tensor(self.boundaries_in_step[ii])
            boundary_right = tf.convert_to_tensor(self.boundaries_in_step[ii + 1])
            predicate = tf.logical_and(tf.greater_equal(global_step_recomp, boundary_left),
                                       tf.less(global_step_recomp, boundary_right))
            pos = tf.cond(pred=predicate, true_fn=lambda: ii, false_fn=lambda: pos)
        return self.learning_rates[pos]

    def get_config(self):
        return {
            'n_boundaries': self.n_boundaries,
            'name': self.name,
        }


class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, max_step_size, name=None):
        super(CosineDecay, self).__init__()
        self.initial_learning_rate = initial_lr
        self.max_step_size = max_step_size
        self.name = name

    def __call__(self, step):
        initial_lr = tf.convert_to_tensor(self.initial_learning_rate)
        pi = tf.convert_to_tensor(np.pi)
        global_step_recomp = tf.cast(step, initial_lr.dtype)
        return initial_lr * 0.5 * (1.0 + tf.cos(global_step_recomp / self.max_step_size * pi))

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'max_step_size': self.max_step_size,
            'name': self.name,
        }


def main():
    import matplotlib.pyplot as plt
    from tqdm import trange

    global_batch_size = 256
    epochs = 5
    n_images = 1281167 * epochs
    initial_lr = 0.03
    lr_decay = 0.1
    lr_decay_boundaries = [1, 3]
    lr_schedule_fn = StepDecay(initial_lr, lr_decay, n_images, epochs, lr_decay_boundaries, global_batch_size)
    optimizer = tf.keras.optimizers.SGD(lr_schedule_fn, momentum=0.9, nesterov=False)

    all_lr = list()
    all_st = list()
    max_step = int(np.ceil(n_images / global_batch_size))
    for step in trange(0, max_step):
        all_st.append(step)
        all_lr.append(optimizer.learning_rate(step))

    plt.plot(all_st, all_lr)
    plt.show()
    return


if __name__ == '__main__':
    main()
