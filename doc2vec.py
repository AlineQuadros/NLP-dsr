# -*- coding: utf-8 -*-

#import gensim.models.word2vec
import pandas as pd
import gensim

# code from: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py

def read_corpus(texts, tokens_only=False):
    for i, line in enumerate(texts):
        try:
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags which is the document index
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        except:
            pass
        
train_corpus = list(read_corpus(clean_dataset.abstract[:10000]))
test_corpus = list(read_corpus(clean_dataset.abstract[40000:41591], tokens_only=True))

def doc2vec_train(train_corpus, vector_size=200, min_count=2, epochs=40):
    model = gensim.models.doc2vec.Doc2Vec(vector_size, min_count, epochs)
    model.build_vocab(train_corpus)
    print(len(model.wv.vocab))
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)


def get_similarity_pairs_word2vec(model):
    for i in range(len(model.docvecs)):
        print(i)
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=10)
        #sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        #rank = [docid for docid, sim in sims].index(doc_id)
        for id, sim in enumerate(sims, 1):
            if sim[1] > 0.4:
                line = "{'ab1':'" + str(clean_dataset.id[i]) + "','ab2':'" + str(clean_dataset.id[sim[0]]) + "','sim_doc2vec':" + str(round(sim[1], 3)) + "}\n"
                with open('data/similarity_doc2vec_10000.csv', 'a') as f:
                    f.write(line)

def assemble_dataset_from_files():
    dataset_doc2vec = pd.DataFrame()
    txt = open('data/similarity_doc2vec_10000.csv', 'r')
    for l in txt:
        l = l.replace("\'", "\"")
        print(l)
        line = json.loads(l)
        dataset_doc2vec = dataset_doc2vec.append(pd.Series([line['ab1'], line['ab2'], line['sim_doc2vec']]), ignore_index=True)
    dataset_doc2vec.columns = ['ab1', 'ab2', 'sim_doc2vec']
    dataset_doc2vec.to_csv("data/similarity_doc2vec_processed.csv", index=False)

