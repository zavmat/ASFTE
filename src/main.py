import pandas as pd
import ast
import en_core_web_sm
from src.Corpus import Corpus
from src.Classifier import Classifier
#%%
#Uncomment to rebuild corpus
#Read crawler output
#seed_pages = pd.read_csv('./data/crawlerinput/ProductFeaturesRaw.csv') #Read crawled seed pages
#help_articles = pd.read_csv('./data/crawlerinput/HelpArticles.csv') # Read crawled help articles
#Build corpus (uncomment to rebuild corpus)
#corpus = Corpus(seed_pages, help_articles, preprocess=True, short_descriptor=False, backtranslation=True) # Build corpus
#Save corpus to csv (uncomment to rebuild corpus)
#corpus.corpus_df.to_csv('./data/corpus/corpus_augmented_backtranslation.csv')

#%% Read corpus
corpus = pd.read_csv('./data/corpus/corpus_augmented_backtranslation.csv')
#%%
#Best parameters derived via grid search
BEST_PARAM = {
    'filter_meta': {'synthetic': 0},  #Use backtranslated feature descriptors
    'gram': 3,  #Use bi-grams
    'max_doc_depth': 5, #Use only up to the 5th level of depth of crawled articles
    'threshold': 0.25, #Set sensitivity/threshold of classifier to 0.25
    'vector_params': {
        'tf_type': 'linear', #Use linear term frequency where the term-frequency equals to the number of occurences
        'apply_idf': True, #Apply inverse document freqency as global weight component
        'idf_type': 'smooth', #The type of inverse document frequency
        'apply_dl': True, #Normalize local(+global) weights by doc length. Important as there is a big variance in doc length
        'dl_type': 'linear'} #Type of document-length scaling to use for weights’ normalization component
}
#%%
#Initialize the classifier
cl = Classifier(corpus)
#%%
# Classify the crawled articles using the best parameter (tuned with grid search)
best_classification_df = cl.classify(**BEST_PARAM)
#%%
def get_classified_only(corpus, classified_docs):


    labeled_only = classified_docs
    labeled_only = labeled_only.dropna() #remove non-classified rows
    labeled_only['id']=labeled_only.index #create column with article path


    #Create a lookup dictionary with articles to classified labels
    lookup_dict = labeled_only[['id','y_pred']].to_dict(orient='records')
    label_values = [{dic['id']: dic['y_pred']} for dic in lookup_dict]
    label_lookup = {}

    for d in label_values:
        label_lookup.update(d)
    label_lookup = dict(label_lookup)

    #Create a list of all articles and seed labels classified
    classified = list(labeled_only['id'].unique())+list(labeled_only['y_pred'].unique())

    #Filter rows of the initial corpus to the ones which were classified (above threshold, not unlabaled)
    classified_corpus_df = corpus[corpus['group_key'].isin(classified)]
    classified_corpus_df.reset_index()
    #Assign classification(product) labels to the articles
    classified_only_df = classified_corpus_df.replace({'group_key':label_lookup})

    return classified_only_df

#%%
labeled_only = get_classified_only(corpus,best_classification_df)

#%%
grouped_texts = labeled_only
grouped_texts = grouped_texts.groupby(['group_key'])['feature_descriptor'].agg(lambda x : ' '.join(x))

#%%
from textacy.ke import textrank, yake, scake

nlp = en_core_web_sm.load()
doc = nlp(grouped_texts[13])
scake(doc, normalize="lower", topn=40)