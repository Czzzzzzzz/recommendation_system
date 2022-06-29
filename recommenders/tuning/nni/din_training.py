import logging
from random import seed
import numpy as np
import os
import pandas as pd
import nni
import sys

if os.path.join('..', '..', '..', 'recommenders') not in sys.path:
    sys.path.append(os.path.join('..', '..', '..', 'recommenders'))

from models.deeprec.models.sequential.din import DIN_RECModel
from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.utils.timer import Timer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("din")


def din_training(params):
    """
    Train DIN using the given hyper-parameters
    """
    logger.debug("Start training...")

    EPOCHS = 10
    BATCH_SIZE = 400

    data_path = os.path.join("..", "..", "..", "tests", "resources", "deeprec", "slirec")
    yaml_file = '../../../recommenders/models/deeprec/config/din.yaml'  

    train_file = os.path.join(data_path, r'train_data')
    valid_file = os.path.join(data_path, r'valid_data')
    test_file = os.path.join(data_path, r'test_data')
    user_vocab = os.path.join(data_path, r'user_vocab.pkl')
    item_vocab = os.path.join(data_path, r'item_vocab.pkl')
    cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
    train_num_ngs = 4
    valid_num_ngs = 4
    test_num_ngs = 9

    hparams = prepare_hparams(yaml_file, 
        embed_l2=0., 
        layer_l2=0., 
        learning_rate=0.001,  # set to 0.01 if batch normalization is disable
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        show_step=20,
        save_model=False,
        MODEL_DIR=os.path.join(data_path, "model/"),
        write_tfevents=True,
        SUMMARIES_DIR=os.path.join(os.environ['NNI_OUTPUT_DIR'], "tensorboard"),
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        cate_vocab=cate_vocab,
        need_sample=True,
        train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
        # enable_BN=params['enable_BN'],
        #activation=params['activation'],
        #dice_momentum=params['dice_momentum']
        #max_seq_length=params['max_seq_length']
        # attention_mode=params['attention_mode']
        max_seq_length=params['max_seq_length']
    )
    input_creator = SequentialIterator
    model = DIN_RECModel(hparams, input_creator, seed=params['seed'])

    with Timer() as train_time:
        model.fit(train_file, valid_file, valid_num_ngs)

    print('Time cost for training is {0:.2f} mins for {1} epochs.'.format(train_time.interval/60.0, EPOCHS))

    logger.debug("Evaluating...")

    metrics_dict = model.run_eval(test_file, num_ngs=test_num_ngs)

    print("current params: ", params)
    print("final result: ", metrics_dict)
    # Report the metrics
    nni.report_final_result(metrics_dict)

    return model


def main(params):
    logger.debug("Args: %s", str(params))
    din_training(params)


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        params = {}

        # in the case of Hyperband, use STEPS to allocate the number of epochs NCF will run for
        if "STEPS" in tuner_params:
            steps_param = tuner_params["STEPS"]
            params["n_epochs"] = int(np.rint(steps_param))
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
