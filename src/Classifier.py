import en_core_web_sm
from TextUtils import clean_text,clean_seed_url,clean_article_url
import textacy
from textacy import make_spacy_doc
from spacy.tokens import Doc
from textacy.representations import Vectorizer
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
#Class to calculate cosine similarity between scraped product features (with labels derived from url structure) and
#unlabeled scraped help articles. The class is first creating a general corpus from all scraped data and select only
#ngrams which have a high tfidf score and appear at least 10 times -> method relevant_corpus_vocabulart
#These features further reduced using a Chi2 test -> select_seed_features
#Finally cosine similarity is calculated between labeled seed words from seed_pages and unlabeled articles.

class Classifier:



    def __init__(self,corpus_df):
        global nlp
        global corpus_textacy
        global group_corpus_textacy
        global common_terms
        nlp = en_core_web_sm.load()
        corpus_textacy = self.df_to_textacy(corpus_df)


    def classify(self, gram, threshold, max_doc_depth=None, vector_params=None, filter_meta=None):
        corpus_textacy_copy = textacy.Corpus(nlp,data = corpus_textacy)
        #Filter corpus by max_doc_depth
        if max_doc_depth is not None:
            corpus_textacy_copy.remove(lambda doc: doc._.meta.get("intra_doc_depth") >= max_doc_depth)
        #Filter any documents by meta
        if filter_meta is not None:
            filtered_corpus = self.subset_corpus_bymeta(corpus_textacy_copy,filter_meta)
            corpus_textacy_copy = textacy.Corpus(nlp, data=filtered_corpus)
        # Group corpus
        group_corpus_textacy = self.group_corpus(corpus_textacy_copy)

        #ngrams for feature extraction
        ngrams = tuple([i + 1 for i in range(gram)])

        #Get corpus common terms (between seeds and articles)
        common_terms = self.get_common_terms(group_corpus_textacy, ngrams)
        #print(common_terms)
        #set_vectorizer
        if vector_params is not None:
            self.set_vectorizer(vocabulary_terms=common_terms,**vector_params)
        else:
            self.set_vectorizer(vocabulary_terms=common_terms)

        #vectorize group corpus
        similarity_mx = self.get_similarity_mx(group_corpus_textacy,ngrams)

        if threshold is not None:
            classified_mx=self.predict(similarity_mx,threshold)
        else:
            classified_mx=self.predict(similarity_mx)

        return classified_mx

    # Method to classify documents based on cosine similarity. Take a threshold value (default=0.2) and select the highest similarity score between seed and target (article) documents
    def predict(self,similarity_mx_df, threshold=0.2):
        similarity_mx_df_threshold = similarity_mx_df[~similarity_mx_df.le(threshold).all(1)]
        similarity_mx_df_threshold['y_pred'] = similarity_mx_df_threshold.idxmax(axis=1)
        predicted = similarity_mx_df.reset_index().merge(similarity_mx_df_threshold, how="outer").set_index("index")
        return predicted

    #Method to vectorize group corpus, and generate a similarity matrix between seed and article vectors
    def get_similarity_mx(self,corpus, ngrams):
        tokenized_docs = (
            doc._.to_terms_list(ngrams=ngrams, normalize="lemma", as_strings=True, filter_nums=True)
            for doc in corpus
        )
        doc_term_mx =  self.vectorizer.fit_transform(tokenized_docs)

        # Get seed and articles pointers from generated csr matrix
        index = 0
        seed_ind = []
        seed_labels = []
        article_ind = []
        article_labels = []
        for doc in corpus:
            if doc._.meta.get("is_seed") == 1:
                seed_ind.append(index)
                seed_labels.append(doc._.meta.get("group_key"))
            else:
                article_ind.append(index)
                article_labels.append(doc._.meta.get("group_key"))
            index += 1

        # Subset csr for seed and articles
        seed_csr = doc_term_mx[seed_ind, :]
        article_csr = doc_term_mx[article_ind, :]

        # Create a similarity matrix between seed documents and article documents
        similarity_mx = cosine_similarity(article_csr, seed_csr)
        # Convert to Dataframe
        similarity_mx_df = pd.DataFrame(similarity_mx, index=article_labels,
                                        columns=seed_labels)


        return similarity_mx_df

    # Group vectorizer to generate vectors for documents in textacy corpus
    def set_vectorizer(self, *args, **kwargs):
        self.vectorizer = Vectorizer(*args, **kwargs)

    #Method to convert from dataframe to textacy corpus
    def df_to_textacy(self,corpus_df):

        corpus_meta = corpus_df.drop(columns=['feature_descriptor','feature_descriptor_clean'], axis=1).to_dict('r')
        corpus_text = corpus_df['feature_descriptor_clean'].tolist()
        records = list(zip(corpus_text, corpus_meta))
        # Create an empty corpus
        corp = textacy.Corpus(lang=nlp)
        # Add all docs to corpus
        for record in records:
            corp.add_record(record)

        return corp


    # Group corpus by label (referrer_url)
    def group_corpus(self,corpus_textacy):
        all_group_keys = list(set([doc.user_data['textacy']['meta']['group_key'] for doc in corpus_textacy.docs]))

        grouped_corp = textacy.Corpus(lang=nlp)
        for group_key in all_group_keys:
            grouped_docs = [doc for doc in corpus_textacy.get(lambda doc: doc._.meta.get('group_key') == group_key)]
            group_doc = make_spacy_doc(Doc.from_docs(grouped_docs))
            meta = {key: grouped_docs[0]._.meta[key] for key in
                    ['group_key', 'is_seed']}  # get meta from one of the docs
            group_doc._.__setattr__('meta', meta)
            grouped_corp.add_doc(group_doc)

        return grouped_corp

    #Subset a dictionary by meta tags provided as a dictionary
    def subset_corpus_bymeta(self,corpus,meta):
        corpus_subset = textacy.Corpus(lang=nlp) #subset to be created
        keys_provided = list(meta.keys()) #keys provided by user to filter on

        # Iterate corpus and
        for doc in corpus:
            #Subset dictionary of each documents meta-tags by the user provided keys
            doc_keys_subset = {key: doc._.meta[key] for key in keys_provided}
            # compare common items. if all match
            if doc_keys_subset == meta :
                # append to filtered docs
                corpus_subset.add_doc(doc)

        return corpus_subset

    # Get vocabulary terms present in both seeds and articles (feature selection)
    def get_common_terms(self,corpus, ngrams):

        # Get article documents
        group_corpus_textacy_articles = textacy.Corpus(lang=nlp)
        group_corpus_textacy_articles = self.subset_corpus_bymeta(corpus, {"is_seed": 0})

        # Get seed documents
        group_corpus_textacy_seed = textacy.Corpus(lang=nlp)
        group_corpus_textacy_seed = self.subset_corpus_bymeta(corpus, {"is_seed": 1})

        # Get seed terms
        terms_seed = []
        for doc in group_corpus_textacy_seed:
            terms_seed.append(
                list(doc._.to_bag_of_terms(ngrams=ngrams, weighting="freq", normalize="lemma", as_strings=True,
                                           filter_nums=True)))
        terms_seed = set(itertools.chain.from_iterable(terms_seed))
        print('# seed terms: '+str(len(terms_seed)))
        # Get article terms
        terms_articles = []
        for doc in group_corpus_textacy_articles:
            terms_articles.append(
                list(doc._.to_bag_of_terms(ngrams=ngrams, weighting="freq", normalize="lemma", as_strings=True,
                                           filter_nums=True)))
        terms_articles = set(itertools.chain.from_iterable(terms_articles))
        print('# help article terms: ' + str(len(terms_articles)))
        # Get common terms
        common_terms = terms_articles.intersection(terms_seed)
        print('# common terms: ' + str(len(common_terms)))
        return list(common_terms)