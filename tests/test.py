import sys

sys.path.append('../recommenders')

from models_pytorch.deeprec.deeprec_utils import prepare_hparams
from datasets.amazon_reviews import download_and_extract, data_preprocessing
from datasets.download_utils import maybe_download

from models_pytorch.deeprec.io.sequential_iterator import SequentialIterator

data_path = os.path.join("resources", "deeprec", "din")
yaml_file = '../recommenders/models/deeprec/config/din.yaml'  

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
    enable_BN=params['enable_BN'],
    activation=params['activation']
    #max_seq_length=params['max_seq_length']
)

print(hparams)

# for test
train_file = os.path.join(data_path, r'train_data')
valid_file = os.path.join(data_path, r'valid_data')
test_file = os.path.join(data_path, r'test_data')
user_vocab = os.path.join(data_path, r'user_vocab.pkl')
item_vocab = os.path.join(data_path, r'item_vocab.pkl')
cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
output_file = os.path.join(data_path, r'output.txt')

reviews_name = 'reviews_Movies_and_TV_5.json'
meta_name = 'meta_Movies_and_TV.json'
reviews_file = os.path.join(data_path, reviews_name)
meta_file = os.path.join(data_path, meta_name)
train_num_ngs = 4 # number of negative instances with a positive instance for training
valid_num_ngs = 4 # number of negative instances with a positive instance for validation
test_num_ngs = 9 # number of negative instances with a positive instance for testing
sample_rate = 0.0001 # sample a small item set for training and testing here for fast example

input_files = [reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab]

if not os.path.exists(train_file):
    download_and_extract(reviews_name, reviews_file)
    download_and_extract(meta_name, meta_file)
    data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs)
    #### uncomment this for the NextItNet model, because it does not need to unfold the user history
    # data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs, is_history_expanding=False)

input_creator = SequentialIterator
model = DIN_RECModel(hparams, input_creator, seed=DEFAULT_SEED)

with Timer() as train_time:
    model.fit(train_file, valid_file, valid_num_ngs)

print('Time cost for training is {0:.2f} mins for {} epochs.'.format(train_time.interval/60.0, EPOCHS))