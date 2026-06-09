"""
neural networks
"""
import tensorflow as tf
import tensorflow_probability as tfp

from .nn_layers import GLOROT_UNIFORM_INITIALIZER, l2_regularizer, mlp_batch_norm, mlp_dense, mlp_dropout

tfd = tfp.distributions

_XAVIER = GLOROT_UNIFORM_INITIALIZER
_L2 = l2_regularizer(0.8)

# VAE/beta-TCVAE networks
def encoder1(x, opt, reuse = False):
	""" encoder network """
	with tf.compat.v1.variable_scope("encoder1", reuse = reuse):

		en_dense1 = mlp_dense(x, opt.inflate_to_size2, "encoder_dense1", _XAVIER)
		en_dense1 = mlp_batch_norm(en_dense1, "encoder_dense1_bn")
		en_dense1 = tf.nn.leaky_relu(en_dense1)
		en_dense1 = mlp_dropout(en_dense1, opt.dropout_rate)

		en_dense2 = mlp_dense(en_dense1, opt.inflate_to_size1, "encoder_dense2", _XAVIER)
		en_dense2 = mlp_batch_norm(en_dense2, "encoder_dense2_bn")
		en_dense2 = tf.nn.relu(en_dense2)
		en_dense2 = mlp_dropout(en_dense2, opt.dropout_rate)

		en_loc = mlp_dense(en_dense2, opt.code_size, "encoder_loc", _XAVIER)

		en_scale = mlp_dense(en_dense2, opt.code_size, "encoder_scale", _XAVIER)
		en_scale = tf.nn.softplus(en_scale)

		return en_loc, en_scale



def decoder2(z, opt, reuse = False):
	""" decoder network """
	with tf.compat.v1.variable_scope("decoder2", reuse=reuse):

		de_dense1 = mlp_dense(z, opt.inflate_to_size1, "decoder_dense1", _XAVIER)
		de_dense1 = mlp_batch_norm(de_dense1, "decoder_dense1_bn")
		de_dense1 = tf.nn.leaky_relu(de_dense1)
		de_dense1 = mlp_dropout(de_dense1, opt.dropout_rate)


		de_dense2 = mlp_dense(de_dense1, opt.inflate_to_size2, "decoder_dense2", _XAVIER)
		de_dense2 = mlp_batch_norm(de_dense2, "decoder_dense2_bn")
		de_dense2 = tf.nn.leaky_relu(de_dense2)
		de_dense2 = mlp_dropout(de_dense2, opt.dropout_rate)

		de_loc = mlp_dense(de_dense2, opt.gex_size, "decoder_loc", _XAVIER)

		de_scale = tf.ones_like(de_loc)


		return tfd.Normal(de_loc, de_scale)


# GAN networks
def generator(z, opt, reuse = False):
	""" generator network """
	with tf.compat.v1.variable_scope("generator", reuse = reuse):

		de_dense1 = mlp_dense(z, opt.inflate_to_size1, "generator_dense1", _XAVIER)
		de_dense1 = mlp_batch_norm(de_dense1, "generator_dense1_bn")
		de_dense1 = tf.nn.leaky_relu(de_dense1)
		de_dense1 = mlp_dropout(de_dense1, opt.dropout_rate)

		de_dense2 = mlp_dense(de_dense1, opt.inflate_to_size2, "generator_dense2", _XAVIER)
		de_dense2 = mlp_batch_norm(de_dense2, "generator_dense2_bn")
		de_dense2 = tf.nn.leaky_relu(de_dense2)
		de_dense2 = mlp_dropout(de_dense2, opt.dropout_rate)

		de_dense3 = mlp_dense(de_dense2, opt.inflate_to_size3, "generator_dense3", _XAVIER)
		de_dense3 = mlp_batch_norm(de_dense3, "generator_dense3_bn")
		de_dense3 = tf.nn.relu(de_dense3)
		de_dense3 = mlp_dropout(de_dense3, opt.dropout_rate)

		de_output = mlp_dense(de_dense3, opt.gex_size, "generator_output", _XAVIER)

		return de_output

def discriminator(x, opt, reuse = False):
	""" discriminator network """
	with tf.compat.v1.variable_scope("discriminator", reuse = reuse):

		disc_dense1 = mlp_dense(x, opt.disc_internal_size1, "disc_dense1", _XAVIER, _L2)
		disc_dense1 = mlp_batch_norm(disc_dense1, "disc_dense1_bn")
		disc_dense1 = tf.nn.leaky_relu(disc_dense1)

		disc_dense2 = mlp_dense(disc_dense1, opt.disc_internal_size2, "disc_dense2", _XAVIER, _L2)
		disc_dense2 = mlp_batch_norm(disc_dense2, "disc_dense2_bn")
		disc_dense2 = tf.nn.leaky_relu(disc_dense2)

		disc_dense3 = mlp_dense(disc_dense2, opt.disc_internal_size3, "disc_dense3", _XAVIER, _L2)
		disc_dense3 = mlp_batch_norm(disc_dense3, "disc_dense3_bn")
		disc_dense3 = tf.nn.relu(disc_dense3)

		disc_output = mlp_dense(disc_dense3, 1, "disc_output", _XAVIER)

		return disc_output, disc_dense3


def mutual_discriminator(x, opt, reuse = False):
	""" InfoGAN discriminator """
	with tf.compat.v1.variable_scope("infogan_discriminator", reuse=reuse):

		disc_dense1 = mlp_dense(x, opt.disc_internal_size1, "disc_dense1", _XAVIER, _L2)
		disc_dense1 = mlp_batch_norm(disc_dense1, "disc_dense1_bn")
		disc_dense1 = tf.nn.leaky_relu(disc_dense1)

		disc_dense2 = mlp_dense(disc_dense1, opt.disc_internal_size2, "disc_dense2", _XAVIER, _L2)
		disc_dense2 = mlp_batch_norm(disc_dense2, "disc_dense2_bn")
		disc_dense2 = tf.nn.leaky_relu(disc_dense2)

		disc_dense3 = mlp_dense(disc_dense2, opt.disc_internal_size3, "disc_dense3", _XAVIER, _L2)
		disc_dense3 = mlp_batch_norm(disc_dense3, "disc_dense3_bn")
		disc_dense3 = tf.nn.relu(disc_dense3)

		disc_output = mlp_dense(disc_dense3, 1, "disc_output", _XAVIER)

		q_dense1 = mlp_dense(disc_dense3, opt.disc_internal_size3, "mutual_dense", _XAVIER)
		q_dense1 = mlp_batch_norm(q_dense1, "mutual_dense_bn")
		q_dense1 = tf.nn.leaky_relu(q_dense1)

		q_output = mlp_dense(
			q_dense1,
			(opt.code_size if opt.InfoGAN_fix_std else opt.code_size * 2),
			"mutual_output",
			_XAVIER,
		)


		return disc_output, q_output

# MichiGAN networks-Conditional GANs with projection discriminator
def con_generator(z, y, opt, reuse = False):
	""" conditional generator network """
	with tf.compat.v1.variable_scope("con_generator", reuse = reuse):
		zy = tf.concat([z, y], axis = 1)
		de_dense1 = mlp_dense(zy, opt.inflate_to_size1, "congen_dense1", _XAVIER)
		de_dense1 = mlp_batch_norm(de_dense1, "congen_dense1_bn")
		de_dense1 = tf.nn.leaky_relu(de_dense1)
		de_dense1 = mlp_dropout(de_dense1, opt.dropout_rate)

		de_dense2 = mlp_dense(de_dense1, opt.inflate_to_size2, "congen_dense2", _XAVIER)
		de_dense2 = mlp_batch_norm(de_dense2, "congen_dense2_bn")
		de_dense2 = tf.nn.leaky_relu(de_dense2)
		de_dense2 = mlp_dropout(de_dense2, opt.dropout_rate)

		de_dense3 = mlp_dense(de_dense2, opt.inflate_to_size3, "congen_dense3", _XAVIER)
		de_dense3 = mlp_batch_norm(de_dense3, "congen_dense3_bn")
		de_dense3 = tf.nn.relu(de_dense3)
		de_dense3 = mlp_dropout(de_dense3, opt.dropout_rate)

		de_output = mlp_dense(de_dense3, opt.gex_size, "congen_output", _XAVIER)

		return de_output


def con_discriminator(x, y, opt, reuse = False):
	""" conditional discriminator network """
	with tf.compat.v1.variable_scope("con_discriminator", reuse = reuse):

		disc_dense1 = mlp_dense(x, opt.disc_internal_size1, "disc_dense1", _XAVIER, _L2)
		disc_dense1 = mlp_batch_norm(disc_dense1, "disc_dense1_bn")
		disc_dense1 = tf.nn.leaky_relu(disc_dense1)

		disc_dense2 = mlp_dense(disc_dense1, opt.disc_internal_size2, "disc_dense2", _XAVIER, _L2)
		disc_dense2 = mlp_batch_norm(disc_dense2, "disc_dense2_bn")
		disc_dense2 = tf.nn.leaky_relu(disc_dense2)

		disc_dense3 = mlp_dense(disc_dense2, opt.disc_internal_size3, "disc_dense3", _XAVIER, _L2)
		disc_dense3 = mlp_batch_norm(disc_dense3, "disc_dense3_bn")
		disc_dense3 = tf.nn.relu(disc_dense3)

		disc_output = mlp_dense(disc_dense3, 1, "disc_output", _XAVIER)

		disc_output1 = disc_output + tf.reduce_sum(y * disc_dense3, axis = 1, keepdims = True)

		return disc_output1, disc_dense3


# wrapping up networks
def vaes_encoder(x, opt, reuse = None):
	with tf.compat.v1.variable_scope("EncoderX2Z", reuse = reuse):
		mu, scale = encoder1(x, opt)
	return mu, scale

def vaes_decoder(z, opt, reuse = None):
	with tf.compat.v1.variable_scope("DecoderZ2X", reuse = reuse):
		h = decoder2(z, opt)
	return h

def gan_generator(z, opt, reuse = None):
	with tf.compat.v1.variable_scope("Generator", reuse = reuse):
		h = generator(z, opt)
	return h

def gan_discriminator(x, opt, reuse = None):
	with tf.compat.v1.variable_scope('Discriminator', reuse = reuse):
		f, f1 = discriminator(x, opt)
	return f, f1

def infogan_discriminator(x, opt, reuse = None):
	with tf.compat.v1.variable_scope('InfoGANDiscriminator', reuse = reuse):
		f, q_out = mutual_discriminator(x, opt)
	return f, q_out

def michigan_generator(z, y, opt, reuse = None):
	with tf.compat.v1.variable_scope("MichiGANGenerator", reuse=reuse):
		h = con_generator(z, y, opt)
	return h

def michigan_discriminator(x, y, opt, reuse = None):
	with tf.compat.v1.variable_scope('MichiGANDiscriminator', reuse = reuse):
		f, f1 = con_discriminator(x, y, opt)
	return f, f1
