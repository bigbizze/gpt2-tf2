import glob
import json
import pickle
import random

import click

from data_pipeline import input_fn
from gpt2_model import *

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

@click.command()
@click.option('--num-layers', type=int, default=24, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=1024, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=16, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=50257, show_default=True, help="Vocab size")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=2, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--graph-mode', type=bool, default=False, show_default=False, help="TF run mode")
@click.option('--distributed', type=bool, default=True, show_default=False, help="distributed training")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          optimizer, batch_size, learning_rate, graph_mode, distributed):
    par_map = {"num_layers": num_layers, "d_model": embedding_size,
               "num_heads": num_heads, "dff": dff,
               "max_seq_len": max_seq_len, "vocab_size": vocab_size}

    # exp_name = "_".join(['{}_{}'.format(k, v) for k, v in par_map.items()])

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(MODEL_DIR + '/model_par.json', 'w') as f:
        json.dump(par_map, f)
    # with open(r"dungeon_tokens.pkl", "rb") as fp:
    #     chunks = pickle.load(fp)
    tf_records = glob.glob((_ROOT + "/data/tf_records/*.tfrecord"))
    train_percent = int(len(tf_records) * (85 / 100))

    print("No. of tf records:- ", len(tf_records))
    train_tf_records = tf_records[:train_percent]
    test_tf_records = tf_records[train_percent:]

    train_dataset = input_fn(train_tf_records, batch_size=batch_size)
    test_dataset = input_fn(test_tf_records, batch_size=batch_size)

    if distributed:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

        with mirrored_strategy.scope():

            model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                         optimizer=optimizer, learning_rate=learning_rate)
            model.load_model("./models/355M/model.ckpt.data-00000-of-00001")
            model.create_optimizer()
            model.create_checkpoint_manager(MODEL_DIR)
            model.create_summary_writer(LOG_DIR)

        model.mirrored_strategy = mirrored_strategy
        model.global_batch_size = tf.cast(batch_size, tf.float32)
    else:
        model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                     optimizer=optimizer, learning_rate=learning_rate)
        model.load_model("./models/355M/model.ckpt.data-00000-of-00001")
        model.create_optimizer()
        model.create_checkpoint_manager(MODEL_DIR)
        model.create_summary_writer(LOG_DIR)

    model.fit([train_dataset, test_dataset], graph_mode)
    print("Training Done................")


if __name__ == "__main__":
    train()
