"""TensorFlow training loops (logic from example_*_runvis.py)."""

from __future__ import annotations

import glob
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tensorflow as tf

from michigan.tf_compat import ensure_v1_graph

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from . import lib
from . import nets

if TYPE_CHECKING:
    from michigan.config import ResolvedTrainingConfig

# Star-style access to match original scripts
vaes_encoder = nets.vaes_encoder
vaes_decoder = nets.vaes_decoder
michigan_generator = nets.michigan_generator
michigan_discriminator = nets.michigan_discriminator

noise_prior = lib.noise_prior
tf_standardGaussian_prior = lib.tf_standardGaussian_prior
c_mutual_mu_var_entropy = lib.c_mutual_mu_var_entropy
sample_X = lib.sample_X
sample_XY = lib.sample_XY
estimate_minibatch_mss_entropy = lib.estimate_minibatch_mss_entropy
compute_con_gp = lib.compute_con_gp


class Options(object):
    def __init__(self, num_cells_train, gex_size, code_size, n_train_epochs, model_path, lambda_tc, lambda_gp, lambda_mi):
        self.num_cells_train = num_cells_train  # number of cells
        self.gex_size = gex_size                 # number of genes
        self.epsilon_use = 1e-16                 # small constant value
        self.n_train_epochs = n_train_epochs     # number of epochs for training
        self.batch_size = 32                     # batch size of the GAN-based methods
        self.vae_batch_size = 128                # batch size of the VAE-based methods
        self.code_size = code_size               # number of codes
        self.noise_size = 118                    # number of noise variables
        self.inflate_to_size1 = 256              # number of neurons
        self.inflate_to_size2 = 512              # number of neurons
        self.inflate_to_size3 = 1024             # number of neurons
        self.TotalCorrelation_lamb = lambda_tc   # hyperparameter for the total correlation penalty in beta-TCVAE
        self.InfoGAN_fix_std = True              # fixing the standard deviation or not for the Q network of InfoGAN
        self.dropout_rate = 0.2                  # dropout hyperparameter
        self.disc_internal_size1 = 1024          # number of neurons
        self.disc_internal_size2 = 512           # number of neurons
        self.disc_internal_size3 = code_size     # number of neurons
        self.num_cells_generate = 3000           # number of sampled cells
        self.GradientPenaly_lambda = lambda_gp   # hyperparameter for the gradient penalty of Wasserstein GANs
        self.latentSample_size = 1               # number of samples of the encoder of VAEs
        self.MutualInformation_lamb = lambda_mi  # hyperparameter for the mutual information penalty in InfoGAN
        self.Diters = 5                          # number of training discriminator network per training of generator network of Wasserstein GANs
        self.model_path = model_path             # path saving the model


def run_beta_tcvae(data_matrix, config: "ResolvedTrainingConfig"):
    """
    Code for training beta-TCVAE (beta = 100) on Tabula Muris heart data
    (from example_beta_TCVAE_runvis.py).
    """
    ensure_v1_graph()
    tf.compat.v1.reset_default_graph()

    opt = Options(data_matrix.shape[0], data_matrix.shape[1], config.code_size, config.n_train_epochs, config.model_path,
                  config.lambda_tc, config.lambda_gp, config.lambda_mi)

    z_v = tf.compat.v1.placeholder(tf.float32, shape = (None, opt.code_size))
    X_v = tf.compat.v1.placeholder(tf.float32, shape = (None, opt.gex_size))

    ## encoder
    z_gen_mean_v, z_gen_std_v = vaes_encoder(X_v, opt)

    ### reparameterization of latent space
    batch_size = tf.shape(z_gen_mean_v)[0]
    eps = tf.compat.v1.random_normal(shape=[batch_size, opt.code_size])
    z_gen_data_v = z_gen_mean_v + z_gen_std_v * eps

    ### latent entropies in a minibatch
    margin_entropy_mss, joint_entropy_mss = estimate_minibatch_mss_entropy(z_gen_mean_v, z_gen_std_v, z_gen_data_v, opt)
    ### total correlation in a minibatch
    TotalCorre_mss = tf.reduce_sum(margin_entropy_mss) - tf.reduce_sum(joint_entropy_mss)

    ## decoder
    z_gen_decoder = vaes_decoder(z_gen_data_v, opt)

    ### generated data
    X_gen_data = z_gen_decoder.sample(opt.latentSample_size)
    X_gen_data = tf.reshape(X_gen_data , tf.shape(X_gen_data)[1:])

    ## loss elements
    ### reconstruction error
    z_gen_de = z_gen_decoder.log_prob(X_v)
    z_gen_de_value = tf.reduce_sum(z_gen_de, [1])
    rec_x_loss = - tf.reduce_mean(z_gen_de_value)

    ### latent prior and posterior probabilities
    stg_prior = tf_standardGaussian_prior(tf.shape(X_v)[0], opt.code_size)
    latent_prior = stg_prior.log_prob(z_gen_data_v)
    latent_posterior = c_mutual_mu_var_entropy(z_gen_mean_v, z_gen_std_v, z_gen_data_v, opt)

    ### latent joint prior and posterior probabilities
    latent_prior_joint = tf.reduce_sum(latent_prior, [1])
    latent_posterior_joint = tf.reduce_sum(latent_posterior, [1])

    ### KL divergence
    kl_latent = - tf.reduce_mean(latent_prior_joint) + tf.reduce_mean(latent_posterior_joint)

    ### VAE/beta-TCVAE loss function
    obj_vae = rec_x_loss  + kl_latent + opt.TotalCorrelation_lamb * TotalCorre_mss

    ## time step
    time_step = tf.compat.v1.placeholder(tf.int32)

    ## training tensors
    tf_all_vars = tf.compat.v1.trainable_variables()
    encodervar  = [var for var in tf_all_vars if var.name.startswith("EncoderX2Z")]
    decodervar  = [var for var in tf_all_vars if var.name.startswith("DecoderZ2X")]

    optimizer_vae = tf.compat.v1.train.AdamOptimizer(1e-4)
    opt_vae = optimizer_vae.minimize(obj_vae, var_list = encodervar + decodervar)

    saver = tf.compat.v1.train.Saver()
    global_step = tf.compat.v1.Variable(0, name = 'global_step', trainable = False, dtype = tf.int32)

    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables_initializer().run()
    assign_step_zero = tf.compat.v1.assign(global_step, 0)
    init_step = sess.run(assign_step_zero)

    x_input = data_matrix.copy()
    index_shuffle = list(range(opt.num_cells_train))
    current_step = 0

    for epoch in range(opt.n_train_epochs):
        # shuffling the data per epoch
        np.random.shuffle(index_shuffle)
        x_input = x_input[index_shuffle, :]

        for i in range(0, opt.num_cells_train // opt.vae_batch_size):

            # train VAE/beta-TCVAE in each minibatch
            x_data = sample_X(x_input, opt.vae_batch_size)
            z_data = noise_prior(opt.vae_batch_size, opt.code_size)
            sess.run([opt_vae], {X_v : x_data, z_v: z_data, time_step : current_step})

            current_step += 1

        obj_vae_value, TC_value = sess.run([obj_vae, TotalCorre_mss], {X_v : x_data, z_v: z_data, time_step : current_step})

        print('epoch: {}; iteration: {}; beta-TCVAE loss:{}; Total Correlation:{}'.format(epoch, current_step, obj_vae_value, TC_value))

    model_file_path = opt.model_path + "/models_tcvae"
    saving_model = saver.save(sess, model_file_path, global_step = current_step)

    x_input = data_matrix.copy()
    z_mean = []
    for i in range(0, opt.num_cells_train, opt.batch_size):
        batch_start = i
        batch_end = i + opt.batch_size
        if batch_end > opt.num_cells_train:
            batch_end = opt.num_cells_train
        z_mean.append(sess.run(z_gen_mean_v, {X_v: x_input[batch_start:batch_end]}))
    z_mean = np.vstack(z_mean)
    z_mean.shape

    embed_path = os.path.expanduser(config.embed_np_write_address)
    np.save(embed_path, z_mean)


def run_michigan_mean(data_matrix, config: "ResolvedTrainingConfig"):
    """
    Code for training MichiGAN with mean representations from beta-TCVAE (beta = 100)
    on Tabula Muris heart data (from example_MichiGAN_mean_runvis.py).
    """
    warnings.filterwarnings('ignore')

    ensure_v1_graph()
    tf.compat.v1.reset_default_graph()

    opt = Options(data_matrix.shape[0], data_matrix.shape[1], config.code_size, config.n_train_epochs, config.model_path,
                  config.lambda_tc, config.lambda_gp, config.lambda_mi)
    path_betatcvae = glob.glob(f"{config.btcvae_model_path}/models_tcvae-*.index")[0][:-len(".index")]

    z_v = tf.compat.v1.placeholder(tf.float32, shape = (None, opt.code_size))
    X_v = tf.compat.v1.placeholder(tf.float32, shape = (None, opt.gex_size))

    ## encoder
    z_gen_mean_v, z_gen_std_v = vaes_encoder(X_v, opt)

    ### reparameterization of latent space
    batch_size = tf.shape(z_gen_mean_v)[0]
    eps = tf.compat.v1.random_normal(shape=[batch_size, opt.code_size])
    z_gen_data_v = z_gen_mean_v + z_gen_std_v * eps

    ### latent entropies in a minibatch
    margin_entropy_mss, joint_entropy_mss = estimate_minibatch_mss_entropy(z_gen_mean_v, z_gen_std_v, z_gen_data_v, opt)
    ### total correlation in a minibatch
    TotalCorre_mss = tf.reduce_sum(margin_entropy_mss) - tf.reduce_sum(joint_entropy_mss)

    ## decoder
    z_gen_decoder = vaes_decoder(z_gen_data_v, opt)

    ### generated data
    X_gen_data = z_gen_decoder.sample(opt.latentSample_size)
    X_gen_data = tf.reshape(X_gen_data , tf.shape(X_gen_data)[1:])

    ## loss elements
    ### reconstruction error
    z_gen_de = z_gen_decoder.log_prob(X_v)
    z_gen_de_value = tf.reduce_sum(z_gen_de, [1])
    rec_x_loss = - tf.reduce_mean(z_gen_de_value)

    ### latent prior and posterior probabilities
    stg_prior = tf_standardGaussian_prior(tf.shape(X_v)[0], opt.code_size)
    latent_prior = stg_prior.log_prob(z_gen_data_v)
    latent_posterior = c_mutual_mu_var_entropy(z_gen_mean_v, z_gen_std_v, z_gen_data_v, opt)

    ### latent joint prior and posterior probabilities
    latent_prior_joint = tf.reduce_sum(latent_prior, [1])
    latent_posterior_joint = tf.reduce_sum(latent_posterior, [1])

    ### KL divergence
    kl_latent = - tf.reduce_mean(latent_prior_joint) + tf.reduce_mean(latent_posterior_joint)

    ### VAE/beta-TCVAE loss function
    obj_vae = rec_x_loss  + kl_latent + opt.TotalCorrelation_lamb * TotalCorre_mss

    ## time step
    time_step = tf.compat.v1.placeholder(tf.int32)

    ## training tensors
    tf_all_vars = tf.compat.v1.trainable_variables()
    encodervar  = [var for var in tf_all_vars if var.name.startswith("EncoderX2Z")]
    decodervar  = [var for var in tf_all_vars if var.name.startswith("DecoderZ2X")]

    optimizer_vae = tf.compat.v1.train.AdamOptimizer(1e-4)
    opt_vae = optimizer_vae.minimize(obj_vae, var_list = encodervar + decodervar)

    saver1 = tf.compat.v1.train.Saver()
    global_step = tf.compat.v1.Variable(0, name = 'global_step', trainable = False, dtype = tf.int32)

    sess1 = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables_initializer().run()
    saver1.restore(sess1,  path_betatcvae)

    z = tf.compat.v1.placeholder(tf.float32, shape = (None, opt.noise_size))
    X = tf.compat.v1.placeholder(tf.float32, shape = (None, opt.gex_size))
    Y = tf.compat.v1.placeholder(tf.float32, shape = (None, opt.code_size))

    ## generator
    X_gen_data = michigan_generator(z, Y, opt)

    ## discriminator
    Dx_real, Dx_hidden_real = michigan_discriminator(X, Y, opt)
    Dx_fake, Dx_hidden_fake = michigan_discriminator(X_gen_data, Y, opt, reuse = True)


    ## discriminator loss
    obj_d_or = tf.reduce_mean(Dx_real) - tf.reduce_mean(Dx_fake)
    gradient_penalty = compute_con_gp(X, X_gen_data, Y, michigan_discriminator, opt)
    obj_d = obj_d_or + opt.GradientPenaly_lambda * gradient_penalty

    ## generator loss
    obj_g = tf.reduce_mean(Dx_fake)


    ## time step
    time_step = tf.compat.v1.placeholder(tf.int32)


    ## training tensors
    tf_all_vars = tf.compat.v1.trainable_variables()
    dvar  = [var for var in tf_all_vars if var.name.startswith("MichiGANDiscriminator")]
    gvar  = [var for var in tf_all_vars if var.name.startswith("MichiGANGenerator")]

    ### Thanks to taki0112 for the TF StableGAN implementation https://github.com/taki0112/StableGAN-Tensorflow
    from .adam_prediction import Adam_Prediction_Optimizer

    opt_g = Adam_Prediction_Optimizer(
        learning_rate=1e-4, beta1=0.9, beta2=0.999, prediction=True, name="MichiGAN_Adam_G"
    ).minimize(obj_g, var_list=gvar)
    opt_d = Adam_Prediction_Optimizer(
        learning_rate=1e-4, beta1=0.9, beta2=0.999, prediction=False, name="MichiGAN_Adam_D"
    ).minimize(obj_d, var_list=dvar)



    saver = tf.compat.v1.train.Saver()
    global_step = tf.compat.v1.Variable(0, name = 'global_step', trainable = False, dtype = tf.int32)

    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables_initializer().run()
    assign_step_zero = tf.compat.v1.assign(global_step, 0)
    init_step = sess.run(assign_step_zero)

    x_input = data_matrix.copy()
    index_shuffle = list(range(opt.num_cells_train))
    current_step = 0

    ## beta-TCVAE mean representations
    representations_mean = sess1.run(z_gen_mean_v, {X_v: data_matrix})
    y_input = representations_mean.copy()

    for epoch in range(opt.n_train_epochs):
        # shuffling the data per epoch
        np.random.shuffle(index_shuffle)
        x_input = x_input[index_shuffle, :]
        y_input = y_input[index_shuffle, :]

        for i in range(0, opt.num_cells_train // opt.batch_size):
            # train discriminator opt.Diters times per training of generator
            for k in range(opt.Diters):

                x_data, y_data = sample_XY(x_input, y_input, opt.batch_size)
                z_data = noise_prior(opt.batch_size, opt.noise_size)

                sess.run([opt_d], {X : x_data, Y: y_data, z: z_data, time_step : current_step})


            # train generator
            x_data, y_data = sample_XY(x_input, y_input, opt.batch_size)
            z_data = noise_prior(opt.batch_size, opt.noise_size)
            sess.run([opt_g], {X : x_data, Y: y_data, z: z_data, time_step : current_step})


            current_step += 1

        obj_d_value, obj_g_value = sess.run([obj_d, obj_g], {X : x_data, Y: y_data, z: z_data, time_step : current_step})

        print('epoch: {}; iteration: {}; generator loss: {}; discriminator loss:{}'.format(epoch, current_step, obj_g_value, obj_d_value))

    model_file_path = opt.model_path + "/models_michigan_mean"
    saving_model = saver.save(sess, model_file_path, global_step = current_step)




    x_input = data_matrix.copy()
    z_mean = []
    for i in range(0, opt.num_cells_train, opt.batch_size):
        batch_start = i
        batch_end = i + opt.batch_size
        if batch_end > opt.num_cells_train:
            batch_end = opt.num_cells_train
        z_mean.append(sess.run(z_gen_mean_v, {X_v: x_input[batch_start:batch_end]}))
    z_mean = np.vstack(z_mean)
    z_mean.shape

    embed_path = os.path.expanduser(config.embed_np_write_address)
    np.save(embed_path, z_mean)
