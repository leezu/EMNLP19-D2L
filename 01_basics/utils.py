from mxnet import gluon
import gluonnlp as nlp


def load_data_imdb(batch_size):
    imdb_train = nlp.data.IMDB('train')
    imdb_test = nlp.data.IMDB('test')

    def tokenize_while_preserving_score(sample):
        sentence, score = sample
        return sentence.split(), score

    imdb_train_tokens_score = imdb_train.transform(tokenize_while_preserving_score)
    imdb_test_tokens_score = imdb_test.transform(tokenize_while_preserving_score)

    def get_first(first, second):
        return first

    imdb_train_tokens = imdb_train_tokens_score.transform(get_first)
    import itertools
    tokens_iter = itertools.chain.from_iterable(imdb_train_tokens)

    token_counts = nlp.data.count_tokens(tokens_iter)
    imdb_vocab = nlp.Vocab(token_counts, min_freq=10)

    length_clip_500 = nlp.data.ClipSequence(500)

    def preprocess(tokens, score):
        indices = imdb_vocab[length_clip_500(tokens)]
        label = int(score > 5)
        return indices, label

    train_dataset = imdb_train_tokens_score.transform(preprocess, lazy=False)
    test_dataset = imdb_test_tokens_score.transform(preprocess, lazy=False)
    
    padding_val = imdb_vocab[imdb_vocab.padding_token]
    pad_tokens = nlp.data.batchify.Pad(axis=0, pad_val=padding_val)
    stack_labels = nlp.data.batchify.Stack(dtype='float32')
    batchify_fn = nlp.data.batchify.Tuple(pad_tokens, stack_labels)

    train_sampler = nlp.data.FixedBucketSampler(
        lengths=train_dataset.transform(lambda x, y: len(x)),
        batch_size=batch_size, shuffle=True)
    test_sampler = nlp.data.FixedBucketSampler(
        lengths=test_dataset.transform(lambda x, y: len(x)),
        batch_size=batch_size, shuffle=True)

    train_dataloader = gluon.data.DataLoader(train_dataset, batchify_fn=batchify_fn, batch_sampler=train_sampler)
    test_dataloader = gluon.data.DataLoader(test_dataset, batchify_fn=batchify_fn, batch_sampler=test_sampler)

    return train_dataloader, test_dataloader, imdb_vocab