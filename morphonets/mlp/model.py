# -*- coding: utf-8 -*-
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.merge import Dot
from keras.layers import Embedding, Reshape, Activation, Input
import time


def get_embedding_layer(model):
    # for i in range(len(model.layers)):
    #     print("layer {}".format(i))
    #     weights = model.layers[i].get_weights()
    #     for j in range(len(weights)):
    #         print("weight {}, {}".format(i, j))
    #         print(weights[j])
    weights = model.layers[1].get_weights()[0]
    return weights


def get(args):
    # get figuration
    # input ID sequence
    w_input = Input(shape=(1,), dtype='int32', name='input')

    # embedding sequence
    w = Embedding(input_dim=args.vocabulary_size,
                  output_dim=args.embedding_size,
                  init='glorot_uniform')(w_input)

    # context
    c_input = Input(shape=(1,), dtype='int32', name='context')
    c = Embedding(input_dim=args.vocabulary_size,
                  output_dim=args.embedding_size,
                  init='glorot_uniform')(c_input)

    # output (cos similarity)
    output_ = Dot(axes=2)([w, c])
    output_ = Reshape((1,), input_shape=(1, 1))(output_)
    output = Activation('sigmoid')(output_)

    # model
    SkipGram_model = Model(inputs=[w_input, c_input], outputs=output)

    # initialize the optimizer
    ADAM_ = Adam(lr=args.lr)

    # compile the SkipGram_model
    SkipGram_model.compile(loss='binary_crossentropy', optimizer=ADAM_)

    # save_sample_image the picture of the SkipGram_model
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          ' Save picture of SkipGram_model architecture')
    plot_model(SkipGram_model, show_shapes=True,
               to_file=args.model_picture)

    # show the information of the SkipGram_model
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Model summary")
    print(SkipGram_model.summary())

    # return the skip gram model
    return SkipGram_model
