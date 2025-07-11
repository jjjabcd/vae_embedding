"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network

"""

#from gpu_utils import pick_gpu_lowest_memory
#gpu_free_number = str(pick_gpu_lowest_memory())

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)

import argparse
import numpy as np
import tensorflow as tf
import yaml
import time
import os

# TensorFlow 2.x에서 Keras import 방식 변경
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Lambda

from chemvae import hyperparameters
from chemvae import mol_utils as mu
from chemvae import mol_callbacks as mol_cb
from chemvae.models import encoder_model, load_encoder
from chemvae.models import decoder_model, load_decoder
from chemvae.models import property_predictor_model, load_property_predictor
from chemvae.models import variational_layers
from functools import partial

# GPU 환경 설정 (TensorFlow 2.x 방식)
print("[INFO] Setting up TensorFlow 2.x environment...")

# XLA JIT 컴파일 비활성화 (CUDA libdevice 문제 해결)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 메시지 감소
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_XLA'] = '0'
os.environ['TF_DISABLE_XLA'] = '1'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/rlawlsgurjh/miniconda3/envs/chemvae'

print("[INFO] TensorFlow version:", tf.__version__)
print("[INFO] XLA JIT compilation disabled")

# TensorFlow JIT 컴파일 완전 비활성화
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})

# XLA 관련 모든 최적화 비활성화
try:
    tf.config.experimental.set_synchronous_execution(True)
except:
    pass

# TensorFlow 2.x 방식으로 GPU 설정 (XLA 비활성화)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 growth 활성화
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Found {len(gpus)} GPU(s): {gpus}")
        print("[INFO] GPU memory growth enabled")
        
        # 더 강력한 XLA 비활성화
        tf.config.experimental.enable_op_determinism()
        # Determinism을 위한 시드 설정
        tf.random.set_seed(42)
        
    except RuntimeError as e:
        print(f"[INFO] GPU setup error: {e}")
else:
    print("[INFO] No GPU found, using CPU")
    
print("[INFO] TensorFlow environment setup completed")


def vectorize_data(params):
    # @out : Y_train /Y_test : each is list of datasets.
    #        i.e. if reg_tasks only : Y_train_reg = Y_train[0]
    #             if logit_tasks only : Y_train_logit = Y_train[0]
    #             if both reg and logit_tasks : Y_train_reg = Y_train[0], Y_train_reg = 1
    #             if no prop tasks : Y_train = []

    MAX_LEN = params['MAX_LEN']

    CHARS = yaml.safe_load(open(params['char_file']))
    params['NCHARS'] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
    #INDICES_CHAR = dict((i, c) for i, c in enumerate(CHARS))

    ## Load data for properties
    if params['do_prop_pred'] and ('train_data_file' in params) and ('test_data_file' in params):
        if "data_normalization_out" in params:
            normalize_out = params["data_normalization_out"]
        else:
            normalize_out = None

            print('reg_tasks: ', params['reg_prop_tasks'])

        ################
        if ("reg_prop_tasks" in params) and ("logit_prop_tasks" in params):
            smiles, Y_reg, Y_logit = mu.load_smiles_and_data_df(params['train_data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], logit_tasks=params['logit_prop_tasks'],
                    normalize_out = normalize_out)
            val_smiles, val_Y_reg, val_Y_logit = mu.load_smiles_and_data_df(params['test_data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], logit_tasks=params['logit_prop_tasks'],
                    normalize_out = normalize_out)
        elif "logit_prop_tasks" in params:
            smiles, Y_logit = mu.load_smiles_and_data_df(params['train_data_file'], MAX_LEN,
                    logit_tasks=params['logit_prop_tasks'], normalize_out=normalize_out)
            val_smiles, val_Y_logit = mu.load_smiles_and_data_df(params['test_data_file'], MAX_LEN,
                    logit_tasks=params['logit_prop_tasks'], normalize_out=normalize_out)
        elif "reg_prop_tasks" in params:
            smiles, Y_reg = mu.load_smiles_and_data_df(params['train_data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], normalize_out=normalize_out)
            val_smiles, val_Y_reg = mu.load_smiles_and_data_df(params['test_data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], normalize_out=normalize_out)
        else:
            raise ValueError("please sepcify logit and/or reg tasks")

    ## Load data if no properties
    else:
        smiles = mu.load_smiles_and_data_df(params['train_data_file'], MAX_LEN)
        val_smiles = mu.load_smiles_and_data_df(params['test_data_file'], MAX_LEN)

    if 'limit_data' in params.keys():
        sample_idx = np.random.choice(np.arange(len(smiles)), params['limit_data'], replace=False)
        smiles=list(np.array(smiles)[sample_idx])
        if params['do_prop_pred'] and ('train_data_file' in params):
            if "reg_prop_tasks" in params:
                Y_reg =  Y_reg[sample_idx]
            if "logit_prop_tasks" in params:
                Y_logit =  Y_logit[sample_idx]

    print('Training set size is', len(smiles))
    print('Test set size is', len(val_smiles))
    print('first smiles: \"', smiles[0], '\"')
    print('total chars:', NCHARS)

    print('Vectorization...')
    X = mu.smiles_to_hot(smiles, MAX_LEN, params['PADDING'], CHAR_INDICES, NCHARS)
    val_X = mu.smiles_to_hot(val_smiles, MAX_LEN, params['PADDING'], CHAR_INDICES, NCHARS)

    print('Total Data size', X.shape[0] + val_X.shape[0])
    if np.shape(X)[0] % params['batch_size'] != 0:
        X = X[:np.shape(X)[0] // params['batch_size'] * params['batch_size']]
        if params['do_prop_pred']:
            if "reg_prop_tasks" in params:
                Y_reg = Y_reg[:np.shape(Y_reg)[0] // params['batch_size']
                      * params['batch_size']]
            if "logit_prop_tasks" in params:
                Y_logit = Y_logit[:np.shape(Y_logit)[0] // params['batch_size']
                      * params['batch_size']]

    if np.shape(val_X)[0] % params['batch_size'] != 0:
        val_X = val_X[:np.shape(val_X)[0] // params['batch_size'] * params['batch_size']]
        if params['do_prop_pred']:
            if "reg_prop_tasks" in params:
                val_Y_reg = val_Y_reg[:np.shape(val_Y_reg)[0] // params['batch_size']
                      * params['batch_size']]
            if "logit_prop_tasks" in params:
                val_Y_logit = val_Y_logit[:np.shape(val_Y_logit)[0] // params['batch_size']
                      * params['batch_size']]


    X_train, X_test = X, val_X
    print(X_train.shape, X_test.shape)
    print('shape of input vector : {}', np.shape(X_train))
    print('Training set size is {}, after filtering to max length of {}'.format(
        np.shape(X_train), MAX_LEN))

    if params['do_prop_pred']:
        # !# add Y_train and Y_test here
        Y_train = []
        Y_test = []
        if "reg_prop_tasks" in params:
            Y_reg_train, Y_reg_test = Y_reg, val_Y_reg
            Y_train.append(Y_reg_train)
            Y_test.append(Y_reg_test)
            print(len(Y_reg_train), len(Y_reg_test))
        if "logit_prop_tasks" in params:
            Y_logit_train, Y_logit_test = Y_reg, val_Y_reg
            Y_train.append(Y_logit_train)
            Y_test.append(Y_logit_test)

        return X_train, X_test, Y_train, Y_test

    else:
        return X_train, X_test


def load_models(params):

    def identity(x):
        return tf.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params['kl_loss_weight'])

    if params['reload_model'] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # Decoder
    if params['do_tgru']:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    x_out = Lambda(identity, name='x_pred')(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params['do_prop_pred']:
        if params['reload_model'] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ) and
                ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 )):

            reg_prop_pred, logit_prop_pred   = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.extend([reg_prop_pred,  logit_prop_pred])

        # regression only scenario
        elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError('no logit tasks or regression tasks specified for property prediction')

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print('x_mean shape in kl_loss: ', x_mean.shape)
    kl_loss = - 0.5 * \
        K.mean(1 + x_log_var - K.square(x_mean) -
              K.exp(x_log_var), axis=-1)
    return kl_loss


def main_no_prop(params):
    start_time = time.time()

    X_train, X_test = vectorize_data(params)
    AE_only_model, encoder, decoder, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(learning_rate=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(learning_rate=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(learning_rate=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)
    callbacks = [ vae_anneal_callback, csv_clb]

    if 'checkpoint_path' in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                params=params, save_best_only=False))

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}

    AE_only_model.compile(loss=model_losses,
        loss_weights=[xent_loss_weight,
          kl_loss_var],
        optimizer=optim,
        metrics={'x_pred': ['categorical_accuracy',vae_anneal_metric]}
        )

    keras_verbose = 1  # params['verbose_print'] - 강제로 1로 설정
    print(f"[INFO] Starting no-property training with {params['epochs']} epochs...")
    print(f"[INFO] Batch size: {params['batch_size']}")
    print(f"[INFO] Training samples: {X_train.shape[0]}")
    print(f"[INFO] Validation samples: {X_test.shape[0]}")
    print("[INFO] Model compilation completed, starting training...")
    print(f"[INFO] Will train from epoch {params['prev_epochs']} to {params['epochs']}")

    AE_only_model.fit(X_train, model_train_targets,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    initial_epoch=params['prev_epochs'],
                    callbacks=callbacks,
                    verbose=keras_verbose,
                    validation_data=[ X_test, model_test_targets]
                    )
                    
    print("[INFO] Training completed! Saving models...")

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')
    return

def main_property_run(params):
    start_time = time.time()
    print(f"[INFO] Starting main_property_run at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # load data
    print("[INFO] Loading and vectorizing data...")
    X_train, X_test, Y_train, Y_test = vectorize_data(params)
    print("[INFO] Data loading completed!")

    # load full models:
    print("[INFO] Building models...")
    AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = load_models(params)
    print("[INFO] Model building completed!")

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(learning_rate=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(learning_rate=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(learning_rate=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    ae_loss_weight = 1. - params['prop_pred_loss_weight']
    model_loss_weights = {
                    'x_pred': ae_loss_weight*xent_loss_weight,
                    'z_mean_log_var':   ae_loss_weight*kl_loss_var}

    prop_pred_loss_weight = params['prop_pred_loss_weight']


    if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
        model_train_targets['reg_prop_pred'] = Y_train[0]
        model_test_targets['reg_prop_pred'] = Y_test[0]
        model_losses['reg_prop_pred'] = params['reg_prop_pred_loss']
        model_loss_weights['reg_prop_pred'] = prop_pred_loss_weight
    if ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
        if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            model_train_targets['logit_prop_pred'] = Y_train[1]
            model_test_targets['logit_prop_pred'] = Y_test[1]
        else:
            model_train_targets['logit_prop_pred'] = Y_train[0]
            model_test_targets['logit_prop_pred'] = Y_test[0]
        model_losses['logit_prop_pred'] = params['logit_prop_pred_loss']
        model_loss_weights['logit_prop_pred'] = prop_pred_loss_weight


    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)

    callbacks = [ vae_anneal_callback, csv_clb]
    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    # control verbose output - 강제로 1로 설정하여 학습 진행 상황 확인
    keras_verbose = 1  # params['verbose_print']
    print(f"[INFO] Starting training with {params['epochs']} epochs...")
    print(f"[INFO] Batch size: {params['batch_size']}")
    print(f"[INFO] Training samples: {X_train.shape[0]}")
    print(f"[INFO] Validation samples: {X_test.shape[0]}")
    print("[INFO] Model compilation completed, starting training...")

    if 'checkpoint_path' in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                params=params, prop_pred_model = property_predictor,save_best_only=False))

    AE_PP_model.compile(loss=model_losses,
               loss_weights=model_loss_weights,
               optimizer=optim,
               metrics={'x_pred': ['categorical_accuracy',
                    vae_anneal_metric]})


    print("[INFO] Starting model training...")
    print(f"[INFO] Will train from epoch {params['prev_epochs']} to {params['epochs']}")
    
    AE_PP_model.fit(X_train, model_train_targets,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         initial_epoch=params['prev_epochs'],
                         callbacks=callbacks,
                         verbose=keras_verbose,
         validation_data=[X_test, model_test_targets]
     )
     
    print("[INFO] Training completed! Saving models...")

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    property_predictor.save(params['prop_pred_weights_file'])

    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default=None)
    args = vars(parser.parse_args())
    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    params = hyperparameters.load_params(args['exp_file'])
    print("All params:", params)

    if args['directory'] is not None:
        #params['train_data_file'] = os.path.join(args['directory'], params['train_data_file'])
        #params['test_data_file'] = os.path.join(args['directory'], params['test_data_file'])
        params['char_file'] = os.path.join(args['directory'], params['char_file'])
        params['encoder_weights_file'] = os.path.join(args['directory'], params['encoder_weights_file'])
        params['decoder_weights_file'] = os.path.join(args['directory'], params['decoder_weights_file'])
        params['prop_pred_weights_file'] = os.path.join(args['directory'], params['prop_pred_weights_file'])
        params['history_file'] = os.path.join(args['directory'], params['history_file'])
        params['checkpoint_path'] = os.path.join(args['directory'], params['checkpoint_path'])

    print("Corrected params", params['checkpoint_path'])

    if params['do_prop_pred'] :
        main_property_run(params)
    else:
        main_no_prop(params)
