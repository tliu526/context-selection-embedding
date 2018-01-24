"""
Data processing and main method for running Blei Lab's context-selection model on medical data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cPickle as pickle
import sys

sys.path.insert(0, 'util/')
from util import config_to_name

sys.path.insert(0, 'model/')
from embedding import fit_emb
from embedding import evaluate_emb
from embedding import dense_array_feeder
from embedding import sparse_array_feeder


def process_seq(infile, vocab_size):
    """
    Takes a .seq infile and converts it to a num_visits * vocab_size matrix, where the ijth entry is the count of 
    the jth code in the vocabulary present in patient i's visit context.
    :param infile: 
    :param vocab_size: 
    :return: 
    """
    seq = pickle.load(open(infile))

    contexts = [context for context, _ in seq if len(set(context)) > 1]

    mat = np.zeros((len(contexts), vocab_size))

    for i, context in enumerate(contexts):
        for code_idx in context:
            mat[i, code_idx] += 1

    print('mean number of codes per context: ', np.mean(np.sum(mat, axis=0)))

    return mat


def embedding_experiment(config, mat, outname):
    np.random.seed(seed=27)

    trainset = mat
    #testset = data['testset']['scores']

    """
        trainset: scores: a sparse matrix, each ij entry is the rating of movie j given by person i, or the count of item j in basket i
        testset:  [same structure as trainset]
    """

    # one can always redefine zie.generate_batch(reviews, rind) to use other format of trainset and testset

    #print('The training set has %d rows and %d columns, and the test set has %d rows' %
    #     (trainset.shape[0], trainset.shape[1], testset.shape[0]))

    # batch_feeder is a function, which will be executed as batch_feeder(trainset[i])
    # its output will be fed into tf place holders
    batch_feeder = sparse_array_feeder

    # fit an emb model
    print('Training set has size: ', trainset.shape)
    emb_model, logg = fit_emb(trainset, batch_feeder, config)
    print('Training done!')

    """
    print('Test set has size: ', testset.shape)
    test_llh = evaluate_emb(testset, batch_feeder, emb_model, config)
    print('Testing done!')
    """
    # Save result
    print('Check result...')
    emb_vec = emb_model['alpha']
    print('Embedding matrix has shape ', emb_vec.shape)

    for k in emb_vec.keys():
        pickle.dump(emb_model[k], open(outname+'_'+k+'.pk', 'wb'), -1)

    print('Done!')


if __name__ == '__main__':

    in_matrix = process_seq('snow_attn_cbow_no_zero.seq', 10574)
    dist = 'poisson'
    max_iter = 500
    nprint = 500

    config = dict(
                  # the dimensionality of the embedding vectors
                  K=100,
                  # the embedding distribution  'poisson' or 'binomial' (N=3)
                  dist=dist,
                  # ratio of negative samples. if there are N0 zeros in one row, only sample (0.1 * N0) from these zero,
                  # it is equivalent to downweight zero-targets with weight 0.1
                  neg_ratio=0.1,
                  # number of optimization iterations
                  max_iter=max_iter,
                  # number of iterations to print objective, training log-likelihood, and validation log-likelihood, and debug values
                  nprint=nprint,
                  # weight for regularization terms of embedding vectors
                  ar_sigma2=1,
                  # uncomment the following line to use the base model
                  #model='base',
                  # uncomment the following line to use context selection. Only the prior 'fixed_bern' works for now
                  model='context_select', prior='fixed_bern', nsample=30, hidden_size=[30, 15], histogram_size=40, nsample_test=1000, selsize=10,
                  )

    print('The configuration is: ')
    print(config)

    embedding_experiment(config, in_matrix, "snow_test")

