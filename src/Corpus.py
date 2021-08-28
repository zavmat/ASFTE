import ast
from itertools import chain

import en_core_web_sm
import pandas as pd
import textacy
import textacy.utils
import textacy.utils
from spacy.tokens import Doc
from textacy import make_spacy_doc
from transformers import MarianMTModel, MarianTokenizer

from TextUtils import clean_text, clean_seed_url, clean_article_url


class Corpus:

    # Initialize corpus from raw seed and articles
    def __init__(self, seed, articles, sample=None, preprocess=None, short_descriptor=None, backtranslation=None):
        self.init_nlp()
        if backtranslation:
            self.target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
            self.target_tokenizer = MarianTokenizer.from_pretrained(self.target_model_name)
            self.target_model = MarianMTModel.from_pretrained(self.target_model_name)
            self.en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
            self.en_tokenizer = MarianTokenizer.from_pretrained(self.en_model_name)
            self.en_model = MarianMTModel.from_pretrained(self.en_model_name)
        self.corpus_textacy = self.build_corpus(seed, articles, sample, preprocess, short_descriptor, backtranslation)
        self.group_corpus_textacy = self.group_corpus()

    def get_textacy_corpus(self):
        return self.corpus_textacy

    def get_textacy_group_corpus(self):
        return self.group_corpus_textacy

    def get_corpus_df(self):
        return self.corpus_df

    # Init Spacy language model
    def init_nlp(self):
        global nlp
        nlp = nlp = en_core_web_sm.load()

        return nlp

    # Build corpus from input seed and articles.
    def build_corpus(self, seed_pages, help_articles, sample, preprocess, short_descriptor, backtranslation):
        print(sample)
        # First normalize and stack data
        if short_descriptor:
            self.corpus_df = self.stack_data(seed_pages, help_articles, short_descriptor)
        else:
            self.corpus_df = self.stack_data(seed_pages, help_articles)
        # Option to take only a sample of data
        if sample:
            self.corpus_df = self.corpus_df.sample(n=sample, random_state=1)
        self.corpus_df = self.corpus_df.reset_index()
        self.corpus_df = self.corpus_df.dropna()

        if backtranslation:
            # filter seed rows
            seeds = self.corpus_df[self.corpus_df.is_seed.eq(True)]
            # We only want to augment data with known labels ==> seeds
            original_descriptors = list(seeds['feature_descriptor'])
            # keep metadata from original corpus (before augmentation)
            original_meta = seeds.drop(columns=['feature_descriptor', 'synthetic'])

            # Create 4 lists for the original descriptors backtranslated from different languages
            synthetic_descriptors1 = self.back_translate(original_descriptors, source_lang="en", target_lang="fr")
            synthetic_descriptors2 = self.back_translate(original_descriptors, source_lang="en", target_lang="es")
            synthetic_descriptors3 = self.back_translate(original_descriptors, source_lang="en", target_lang="it")
            synthetic_descriptors4 = self.back_translate(original_descriptors, source_lang="en", target_lang="pt")

            # construct dataframes by merging original metadata with backtranslated descriptors
            synthetic_corpus1 = original_meta.copy()
            synthetic_corpus1['feature_descriptor'] = synthetic_descriptors1
            synthetic_corpus1['synthetic'] = 1
            synthetic_corpus2 = original_meta.copy()
            synthetic_corpus2['feature_descriptor'] = synthetic_descriptors2
            synthetic_corpus2['synthetic'] = 1
            synthetic_corpus3 = original_meta.copy()
            synthetic_corpus3['feature_descriptor'] = synthetic_descriptors3
            synthetic_corpus3['synthetic'] = 1
            synthetic_corpus4 = original_meta.copy()
            synthetic_corpus4['feature_descriptor'] = synthetic_descriptors4
            synthetic_corpus4['synthetic'] = 1

            # concatenate the dataframes to a single corpus (by this generating 4x more data)
            self.corpus_df = pd.concat(
                [self.corpus_df, synthetic_corpus1, synthetic_corpus2, synthetic_corpus3, synthetic_corpus4])

        # Preprocess text
        if preprocess:
            self.corpus_df['feature_descriptor_clean'] = self.corpus_df['feature_descriptor'].apply(
                lambda x: clean_text(x))
        # Separate text and meta-data
        corpus_meta = self.corpus_df.drop('feature_descriptor', axis=1).to_dict('r')

        if preprocess:
            corpus_text = self.corpus_df['feature_descriptor_clean'].tolist()
        else:
            corpus_text = self.corpus_df['feature_descriptor'].tolist()
        # Zip to a tuple of text and meta
        records = list(zip(corpus_text, corpus_meta))

        # Create an empty corpus
        corp = textacy.Corpus(lang=nlp)

        # Add all docs to corpus
        for record in records:
            corp.add_record(record)

        return corp

    # Group corpus by label (referrer_url)
    def group_corpus(self):
        groups = self.get_groups()
        all_group_keys = list(chain(*(group.keys() for group in groups)))

        grouped_corp = textacy.Corpus(lang=nlp)
        for group_key in all_group_keys:
            grouped_docs = [doc for doc in
                            self.corpus_textacy.get(lambda doc: doc._.meta.get('group_key') == group_key)]
            group_doc = make_spacy_doc(Doc.from_docs(grouped_docs))
            meta = {key: grouped_docs[0]._.meta[key] for key in
                    ['group_key', 'is_seed']}  # get meta from one of the docs
            group_doc._.__setattr__('meta', meta)
            grouped_corp.add_doc(group_doc)

        return grouped_corp

    # Transform raw data from crawlers and stack them together
    def stack_data(self, seed, articles, short_descriptor=None):
        # seed=seed.reset_index()
        # articles=articles.reset_index()
        # Transform seed_pages (crawled from product website)
        seed = seed[["url", "feature_name", "feature_description"]]
        seed.dropna()

        # Concatenate the feature names and their decription
        seed["feature_descriptor"] = seed["feature_name"] + ". " + seed["feature_description"]
        seed['feature_short'] = seed['feature_name']
        seed.dropna(how='any', inplace=True)
        # Use transformed url (only containing the product line) as key
        seed['group_key'] = seed['url'].apply(lambda x: clean_seed_url(x))
        seed['is_seed'] = 1
        seed['intra_doc_depth'] = 1
        seed['synthetic'] = 0
        seed.drop(columns=['feature_name', 'feature_description'], inplace=True)

        # Transform help articles
        # Use the landing page (referrer_url) of article collections as key
        articles['referrer_url'] = articles['referrer_url'].apply(lambda x: clean_article_url(x))
        articles = articles.drop(columns=['data_source'])
        articles['feature_short'] = articles['plain_text'].apply(lambda x: x.split('.')[0])

        # If short descriptor, keep only the short text. Else entire feature description.
        articles = articles.rename(columns={'plain_text': 'feature_descriptor', 'referrer_url': 'group_key'})
        articles['is_seed'] = 0
        articles['synthetic'] = 0

        # Stack dataframes
        stacked_data = pd.concat([articles, seed], axis=0, ignore_index=True)
        if short_descriptor:
            stacked_data = stacked_data.drop(columns=['feature_descriptor'])
            stacked_data = stacked_data.rename(columns={'feature_short': 'feature_descriptor'})

        stacked_data = stacked_data.drop(columns=['feature_short'])
        return stacked_data

    # Get all unique group_keys from corpus along with their is_seed value
    def get_groups(self):
        # Get the group and whether it is seed from document meta
        doc_groups = ({doc._.meta['group_key']: doc._.meta['is_seed']} for doc in self.corpus_textacy)
        # Remove duplicates and remove unique (by group_key)
        unique_groups = [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in doc_groups])]
        return list(unique_groups)

    ## Backtranslation methods adopted from: https://amitness.com/back-translation/

    def translate(self, texts, model, tokenizer, language="fr"):
        # Prepare the text data into appropriate format for the model
        template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
        src_texts = [template(text) for text in texts]

        # Tokenize the texts
        encoded = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors="pt")

        # Generate translation using model
        translated = model.generate(**encoded)

        # Convert the generated tokens indices back into text
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

        return translated_texts

    def back_translate(self, texts, source_lang="en", target_lang="fr"):
        # Translate from source to target language
        fr_texts = self.translate(texts, self.target_model, self.target_tokenizer,
                                  language=target_lang)

        # Translate from target language back to source language
        back_translated_texts = self.translate(fr_texts, self.en_model, self.en_tokenizer,
                                               language=source_lang)

        return back_translated_texts
