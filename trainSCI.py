import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,3"

from model.data_utils import Dataset
from model.models import HANNModel
from model.config import Config
import argparse
#import os

parser = argparse.ArgumentParser()

def main():
    # create instance of config
    config = Config(parser)
    #print("initiating the HANN model")
    #print("value of config restore is {}".format(config.restore))
    # build model
    model = HANNModel(config)
    #print("before building the model")
    model.build()
    #if config.restore:
    #    model.restore_session("results/test/model.weights/") # optional, restore weights
    #print("after building the model")
    config.filename_wordvec ='/home/rxr5423/BERT/BERTGit/bert/vocab_words_SciBERT_fine.txt'
    print("The word Embeddings used are: {}".format(config.filename_wordvec))
    print("overridding the config values: restore {}, accuracy {}".format(config.restore, config.train_accuracy))
    config.train_accuracy = True
    config.restore = False
    print("overriden values: restore {}, accuracy {}".format(config.restore, config.train_accuracy))
    # create datasets
    dev   = Dataset(config.filename_dev, config.processing_word,
                    config.processing_tag, config.max_iter)
    train = Dataset(config.filename_train, config.processing_word,
                    config.processing_tag, config.max_iter)
    test  = Dataset(config.filename_test, config.processing_word,
                    config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

    # evaluate model
    model.restore_session(config.dir_model)
    metrics = model.evaluate(test)

    with open(os.path.join(config.dir_output, 'test_results.txt'), 'a') as file:
        file.write('{}\n'.format(metrics['classification-report']))
        file.write('{}\n'.format(metrics['confusion-matrix']))
        file.write('{}\n\n'.format(metrics['weighted-f1']))

if __name__ == "__main__":
    main()
