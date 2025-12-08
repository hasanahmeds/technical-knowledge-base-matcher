import time
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from data import dataset_type
from data.dataset import Dataset
from models import model_type
from models.model_type import MODELS
from utils.batch_helper import BatchHelper
from utils.config_helpers import MainConfig
from utils.data_utils import DatasetVectorizer
from utils.log_saver import LogSaver
from utils.model_evaluator import ModelEvaluator
from utils.model_saver import ModelSaver
from utils.other_utils import timer, set_visible_gpu, init_config

log = tf.logging.info


def create_experiment_name(model_name, main_config, model_config):
    experiment_name = '{}_{}'.format(model_name, main_config['PARAMS']['embedding_size'])
    if model_name == model_type.ModelType.rnn.name:
        experiment_name += ("_" + model_config['PARAMS']['cell_type'])
    
    experiment_name += ("_" + main_config['PARAMS']['loss_function'])
    
    return experiment_name


def train(
        main_config,
        model_config,
        model_name,
        experiment_name,
        dataset_name,
):
    main_cfg = MainConfig(main_config)
    model = MODELS[model_name]
    dataset = dataset_type.get_dataset(dataset_name)
    
    train_data = dataset.train_set_pairs()
    vectorizer = DatasetVectorizer(
        model_dir=main_cfg.model_dir,
        char_embeddings=main_cfg.char_embeddings,
        raw_sentence_pairs=train_data,
    )
    
    dataset_helper = Dataset(vectorizer, dataset, main_cfg.batch_size)
    max_sentence_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size
    
    train_mini_sen1, train_mini_sen2, train_mini_labels = dataset_helper.pick_train_mini_batch()
    train_mini_labels = train_mini_labels.reshape(-1, 1)
    
    test_sentence1, test_sentence2 = dataset_helper.test_instances()
    test_labels = dataset_helper.test_labels()
    test_labels = test_labels.reshape(-1, 1)
    
    num_batches = dataset_helper.num_batches
    model = model(
        max_sentence_len,
        vocabulary_size,
        main_config,
        model_config,
    )
    model_saver = ModelSaver(
        model_dir=main_cfg.model_dir,
        model_name=experiment_name,
        checkpoints_to_keep=main_cfg.checkpoints_to_keep,
    )
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=main_cfg.log_device_placement,
    )
    
    with tf.Session(config=config) as session:
        global_step = 0
        init = tf.global_variables_initializer()
