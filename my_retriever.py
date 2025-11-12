
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): #term_weighting = tf,tfidf, binary ,etc
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    

    def binary_weighting(self, tf_freq): 
        binary_results = {}
        for doc_id in tf_freq:
            for word, tf in tf_freq[doc_id].items():
                if tf >= 1:
                    binary_results[doc_id] = 1
        return binary_results
    
    def tf_weighting(self, tf_freq):
        best_tf = {}
        for doc_id in tf_freq:
            max_tf = 0
            for word, tf in tf_freq[doc_id].items():
                if tf > max_tf:
                    max_tf = tf
            best_tf[doc_id] = max_tf

        return {}

    def tfidf_weighting(self, tf_freq):
        return {}
    
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        # print(query) #['list', 'articl', 'el', 'ecl', 'el', 'el', 'don', 'rememb']
        # term_count = {} #[{'list':1,'articl':1,'el':3,'ecl':1,'don':1,'rememb':1}]
        
        # #convert preprocessed text into term frequency dictionary
        # for term in query: 
        #     if term not in term_count:
        #         term_count[term] = 1
        #     else:
        #         term_count[term] += 1

        doc_words = self.index.keys() #wallac
        # doc_words_values = self.index.values() #{1604: 1, 1892: 1, 2069: 1, 2569: 1, 2847: 2, 3046: 2, 3098: 1}

        tf_freq = {} # {doc_id: {word: tf_value}}
        for term in query:
            if term in doc_words:
                postings = self.index[term] # {1604: 1, 1892: 1, 2069: 1, 2569: 1, 2847: 2, 3046: 2, 3098: 1}
                for doc_id, count in postings.items():
                    if doc_id not in tf_freq:
                        tf_freq[doc_id] = {}
                    tf_freq[doc_id][term] = count

        print(tf_freq)
        #term counts are term vector/weight 
        if self.term_weighting == 'binary':
            return self.binary_weighting(tf_freq)
        elif self.term_weighting == 'tf':
            return self.tf_weighting(tf_freq)
        elif self.term_weighting == 'tfidf':
            return self.tfidf_weighting(tf_freq)
        else:
            print("Unknown term weighting scheme.")
            return []
        

