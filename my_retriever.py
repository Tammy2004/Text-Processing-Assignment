
import math
from collections import Counter


class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): #term_weighting = tf,tfidf, binary ,etc
        self.index = index #index = {term: {doc_id: tf, doc_id: tf, ...}, ...}
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.doc_norms = {"binary": {}, "tf": {}, "tfidf": {}}
        self.sqrt_doc_norms = {"binary": {}, "tf": {}, "tfidf": {}}
        self.idf = {}

        # IDF precomputation
        for term in index:
            df = len(index[term])
            idf = math.log10((self.num_docs)/df)
            self.idf[term] = idf

        # precomputing ||D|| based on term weighting
        for term, postings in self.index.items():
            for doc_id, tf in postings.items():
                self.doc_norms["binary"][doc_id] = self.doc_norms["binary"].get(doc_id, 0) + 1
                self.doc_norms["tf"][doc_id] = self.doc_norms["tf"].get(doc_id, 0) + math.pow(1 + math.log10(tf), 2)
                self.doc_norms["tfidf"][doc_id] = self.doc_norms["tfidf"].get(doc_id, 0) + math.pow( ( (1 + math.log10(tf)) * self.idf[term]), 2)

        for weighting in self.doc_norms:
            for doc_id, norm in self.doc_norms[weighting].items():
                self.sqrt_doc_norms[weighting][doc_id] = math.sqrt(norm)
        

        
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    

    def binary_weighting(self, tf_freq): 
        # Binary weighting: 1 if term present in document, 0 otherwise
        # Returns: {doc_id: {term: 1 or 0}}
        # tf_freq: {doc_id: {term: tf}}, tf_freq.items() = list of (doc_id, term and tf) pairs
        binary_results = {}
        for doc_id, word_dict in tf_freq.items():
            binary_results[doc_id] = {}
            for word, tf in word_dict.items(): #term and tf pairs
                binary_results[doc_id][word] = 1  #since term is present, weight = 1
        return binary_results #returns doc vectors with binary weights
    
    def tf_weighting(self, tf_freq):
        # TF weighting -- returns the same thing as tf_freq - consider replacing with tf_freq directly or
        # Returns: {doc_id: {term: tf}}
        tfw_results = {}
        for doc_id, word_dict in tf_freq.items():
            tfw_results[doc_id] = {}
            for word, tf in word_dict.items():
                if tf > 0:
                    tfw_results[doc_id][word] = 1 + math.log10(tf)  # weight = raw term frequency

        return tfw_results
    
    def tfidf_weighting(self, tf_freq):
        # TF-IDF weighting: tf * idf for each term
        # Returns: {doc_id: {term: tf*idf}}

        idf_cache = self.idf  # use precomputed idf values
        
        # Compute TF-IDF weights for each document
        tfidf_results = {}
        for doc_id, word_dict in tf_freq.items():
            tfidf_results[doc_id] = {}
            for word, tf in word_dict.items():
                if tf > 0:
                    tf_normalized = 1 + math.log10(tf)
                idf = idf_cache.get(word, 0.0)
                tfidf_results[doc_id][word] = tf_normalized * idf
        
        return tfidf_results
    
    def cosine_similarity(self, query, doc_vectors, weighting_scheme):
        # Compute cosine similarity between query and documents
        # Input: query (list of terms), doc_vectors {doc_id: {term: weight}}
        # Output: {doc_id: cosine_similarity_score}
        
        # Step 1: Build query vector based on weighting scheme
        query_counts = Counter(query)
        query_vector = {}
        
        if weighting_scheme == 'binary':
            for term in query_counts:
                query_vector[term] = 1
        
        elif weighting_scheme == 'tf':
            for term, tf in query_counts.items():
                query_vector[term] = 1 + math.log10(tf)
        
        elif weighting_scheme == 'tfidf':
            for term, qtf in query_counts.items():
                if qtf > 0:
                    tf_normalized = 1 + math.log10(qtf)
                idf = self.idf.get(term, 0.0)  # use precomputed idf
                query_vector[term] = tf_normalized * idf  # tf*idf weighting
        
        # Step 2: Compute query vector norm - summation of Qi squared
        query_norm = 0.0
        for weight in query_vector.values():
            query_norm += weight ** 2
        query_norm = math.sqrt(query_norm) if query_norm > 0 else 1.0
        
        # Step 3: Compute cosine similarity for each document
        cosine_results = {}
        for doc_id, doc_weights in doc_vectors.items():
            # Dot product: sum of (query_weight * doc_weight) for overlapping terms
            # sparse numerator

            dot_product = 0.0
            for term, query_weight in query_vector.items():
                if term in doc_weights:
                    dot_product += query_weight * doc_weights[term]
            
            # Document vector norm - summation of Di squared
            doc_norm = self.sqrt_doc_norms[weighting_scheme].get(doc_id, 1.0)
            
            # Cosine similarity
            if query_norm > 0 and doc_norm > 0:
                cosine = dot_product / (query_norm * doc_norm)
            else:
                cosine = 0.0
            
            cosine_results[doc_id] = cosine
        
        return cosine_results
    
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        # Build tf_freq: {doc_id: {term: tf}} for all query terms
        tf_freq = {}
        query_counts = Counter(query)

        # Iterate unique query terms from the counter
        for term, qf in query_counts.items():
            term_postings = self.index.get(term, {})
            if not term_postings:
                continue
            for doc_id, count in term_postings.items():
                if doc_id not in tf_freq:
                    tf_freq[doc_id] = {}
                tf_freq[doc_id][term] = count

        # If no documents match, return empty list
        if not tf_freq:
            return []
        
        # Step 1: Apply term weighting scheme
        if self.term_weighting == 'binary':
            doc_vectors = self.binary_weighting(tf_freq)
        elif self.term_weighting == 'tf':
            doc_vectors = self.tf_weighting(tf_freq)
        elif self.term_weighting == 'tfidf':
            doc_vectors = self.tfidf_weighting(tf_freq)
        else:
            print("Unknown term weighting scheme.")
            return []

        # Step 2: Rank documents using cosine similarity
        scores = self.cosine_similarity(query, doc_vectors, self.term_weighting)

        # Step 3: Return ranked list of doc ids (descending by score)
        ranked_docs = sorted(scores.keys(), key=lambda doc_id: -scores[doc_id])
        return ranked_docs
    
