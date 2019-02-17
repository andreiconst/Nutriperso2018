#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:02:46 2018

@author: andrei
"""

from __future__ import division
import os
import tensorflow as tf
import numpy as np

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass


class GloVeModel():
    def __init__(self, product_names, embedding_size, cooccurrence_cap = 100,
                 scaling_factor=3/4, batch_size=512, learning_rate=0.05):
        self.embedding_size = embedding_size

        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = product_names
        self.__word_to_id = None
        self.__cooccurrence_matrix = None
        self.__embeddings = None
        self.cooccurrence_cap = cooccurrence_cap
        self.__bias = None
        

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            tf.set_random_seed(123)

            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 0.5, -0.5),
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 0.5, -0.5),
                name="context_embeddings")

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 0.5, -0.5),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 0.5, -0.5),
                                         name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)

            weighting_factor = tf.minimum(
                1.0, 
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor)
                )

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(1 + tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                 tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("GloVe_loss", self.__total_loss)
            
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")
            self.__combined_bias = tf.add(focal_biases, context_biases,
                                                name="combined_bias")


    def train(self, num_epochs, co_occurence, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
#        batches = self.__prepare_batches()
        self.__build_graph()

        with tf.Session(graph=self.__graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            if should_write_summaries:
                print("Writing TensorBoard summaries to {}".format(log_dir))
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            tf.global_variables_initializer().run()
            for epoch in range(num_epochs):
                total_steps = 0

#                shuffle(batches)
                for i in range(int(len(co_occurence.row) /self.batch_size)):
#                    i_s, j_s, counts = batch
                    index = [i + j * int(np.floor(len(co_occurence.row) /self.batch_size)) for j in range(self.batch_size)]
                    i_s = tuple(co_occurence.row[index])
                    j_s = tuple(co_occurence.col[index])
                    counts = tuple(co_occurence.data[index])
                    
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cooccurrence_count: counts}
                    _, loss_val = session.run([self.__optimizer, self.__total_loss], feed_dict=feed_dict)
                    
                    if should_write_summaries and (total_steps + 1) % summary_batch_interval == 0:
                        summary_str = session.run(self.__summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, total_steps)
                    total_steps += 1
                    if total_steps % 1000 == 0:
                        print(np.float(total_steps / int(len(co_occurence.row) /self.batch_size)))
                        print(loss_val)
                if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                    current_embeddings = self.__combined_embeddings.eval()
                    output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                    self.generate_tsne(output_path, embeddings=current_embeddings)
                print(epoch)

            self.__embeddings = self.__combined_embeddings.eval()
            self.__bias = self.__combined_bias.eval()
            if should_write_summaries:
                summary_writer.close()

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    @property    
    def bias(self):
        if self.__bias is None:
            raise NotTrainedError("Need to train model before accessing bias")
        return self.__bias

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)



def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)
