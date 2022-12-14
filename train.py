import os

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from model import evaluate, evaluate_float
from model import srgan
from model import laplacian_loss, gaussian_loss, evaluate_bnn
from model import normalize, normalize_bnn

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr',
                 nbit=16,
                 fn_kernel=None):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()
        if fn_kernel is not None:
            self.kernel = np.load(fn_kernel)
        else:
            self.kernel = None
        
        print(model.summary())

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps,
              evaluate_every=1000, save_best_only=False, nbit=16):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
#            lr = tf.cast(lr, tf.float32)

#            lr = tf.image.adjust_gamma(lr, 0.5)
#            print(tf.math.reduce_max(lr),tf.math.reduce_min(lr))
            loss = self.train_step(lr, hr)
            loss_mean(loss)
            

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset, nbit=nbit)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    def kernel_loss(self, sr, lr):
        lr_estimate = signal.fftconvolve(sr.numpy(), self.kernel, mode='same')
        #lr_estimate = tf.nn.conv2d(sr, kernel, strides=[1, 1, 1, 1], padding='VALID')

        print(lr.shape, lr_estimate[2::4, 2::4].shape)
        exit()

    #     return self.loss(lr, lr_estimate)

    # @tf.function
    # def train_step(self, lr, hr):
    #     with tf.GradientTape() as tape:
    #         lr = tf.cast(lr, tf.float32)
    #         hr = tf.cast(hr, tf.float32)
    #         sr = self.checkpoint.model(lr, training=True)
    #         loss_value = self.loss(hr, sr)

    #     gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
    #     self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

    #     return loss_value

    @tf.function
    def train_step(self, lr, hr, gg=1.):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            # lr = tf.image.adjust_gamma(lr,0.9)
            # hr = tf.image.adjust_gamma(hr,0.9)
            sr = self.checkpoint.model(lr, training=True)
            # sr_ = sr - tf.reduce_min(sr)
            # hr_ = hr - tf.reduce_min(hr)
            # loss_value = self.loss(sr, hr)            
            loss_value = self.loss(sr, hr)            
            

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value


    def evaluate(self, dataset, nbit=16):
        # return evaluate(self.checkpoint.model, dataset, nbit=nbit)
        return evaluate_float(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class EdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class WdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]), nbit=16, fn_kernel=None):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir,fn_kernel=fn_kernel)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class BNNTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]),
                 nbit=16,
                 fn_kernel=None):
        super().__init__(model, loss=laplacian_loss, learning_rate=learning_rate, checkpoint_dir=checkpoint_dir,fn_kernel=fn_kernel)

    def evaluate(self, dataset, nbit=16):
        return evaluate_bnn(self.checkpoint.model, dataset, nbit=nbit)

    @tf.function
    def train_step(self, lr, hr, gg=1.):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)

            mu, s, ae, l = self.loss(sr, hr)
            loss_value = tf.reduce_mean(l, axis=(1,2))

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        # return mu, s, ae, l
        return loss_value

    def train(self, train_dataset, valid_dataset, steps,
              evaluate_every=1000, save_best_only=False, nbit=16):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            oghr = tf.identity(hr)
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            # print(tf.math.reduce_min(lr))
            # print(tf.math.reduce_max(lr))
            # print(tf.math.reduce_min(hr))
            # print(tf.math.reduce_max(hr))

            loss_value = self.train_step(lr, hr)
            loss_mean(loss_value)

            # mu, s, ae, l = self.train_step(lr, hr)

            # print(f'\nstep {step}:')
            # print(f'  Num non-finite: {tf.math.reduce_sum(tf.cast(~tf.math.is_finite(l), tf.float32))}')
            # print(f'  Min:  {tf.math.reduce_min(l)}')
            # print(f'  Max:  {tf.math.reduce_max(l)}')

            # # lterm = ae * tf.math.exp(-s)
            # # lterm = ae / (tf.math.log(s + 1e-7))
            # # print(f'  lterm Num non-finite: {tf.math.reduce_sum(tf.cast(~tf.math.is_finite(lterm), tf.float32))}')
            # # print(f'  lterm min:  {tf.math.reduce_min(lterm)}')
            # # print(f'  lterm max:  {tf.math.reduce_max(lterm)}')

            # fig, axs = plt.subplots(1, 7,figsize=(12.0, 3.0))
            # im0 = axs[0].imshow(hr[0])
            # axs[0].set_title('GT')
            # fig.colorbar(im0, ax=axs[0])

            # im1 = axs[1].imshow(lr[0])
            # axs[1].set_title('Input')
            # fig.colorbar(im1, ax=axs[1])

            # im2 = axs[2].imshow(mu[0])
            # axs[2].set_title('$\\mu$')
            # fig.colorbar(im2, ax=axs[2])

            # im3 = axs[3].imshow(s[0])
            # axs[3].set_title('$\\sigma$')
            # fig.colorbar(im3, ax=axs[3])

            # im4 = axs[4].imshow(ae[0])
            # axs[4].set_title('$|\\mu - GT|$')
            # fig.colorbar(im4, ax=axs[4])

            # im5 = axs[5].imshow(l[0])
            # axs[5].set_title('$\\mathcal{L}$')
            # fig.colorbar(im5, ax=axs[5])

            # im6 = axs[6].imshow(oghr[0])
            # axs[6].set_title('OG GT')
            # fig.colorbar(im6, ax=axs[6])

            # plt.tight_layout()
            # plt.savefig(f'loss_fig/step_{step}.png')
            # plt.close()

            # loss_value = tf.math.reduce_mean(l)
            # print(f'  loss: {loss_value}')

            # loss_mean(l)
            

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset, nbit=nbit)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()



class SrganGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        if content_loss == 'VGG22':
            self.vgg = srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.content_loss = content_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_dataset, steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        for lr, hr in train_dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                pls_metric.reset_states()
                dls_metric.reset_states()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
