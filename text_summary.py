import spacy
import numpy as np
import collections

try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    print("Please download and install 'en_core_web_md' correctly.")




class KeywordExtractor:
    def __init__(self, damping_coeff = 0.85, error_thresold = 1e-5, epochs = 50) -> None:
        '''
        ## This KeywordExtractor is based on TextRank algorithm inspired from PageRank algorithm. 
        ### damping_coeff: float 
        This is the value by which the current iteration of calculating weights will be affected w.r.t. the previous iteration.
        ### error_thresold: float
        This is the minimum difference between the weights in to epochs of weight calculation. Below which the calculation will stop.
        ### epochs: int
        The number of iterations the weights will be updated if not converged.
        '''
        self.d = damping_coeff
        self.error_thresold = error_thresold
        self.token_weights = None
        self.epochs = epochs
    
    def getSentTokens(self, document, pos_to_consider, lower, custom_stopwords):
        sentences = []
        for sent in document.sents:
            sent_token = []
            for token in sent:
                if token.is_stop or token.text in custom_stopwords or token.is_punct or token.pos_ not in pos_to_consider:
                    continue
                if lower:
                    sent_token.append(token.text.lower())
                else:
                    sent_token.append(token.text.lower())
            sentences.append(sent_token)
        return sentences
    
    def createVocab(self, segments : list[list[str]]):
        vocab = {}
        i = 0
        for segment in segments:
            for word in segment:
                if word not in vocab.keys():
                    vocab[word] = i
                    i+=1
        return vocab
    
    def createPairs(self, segments, sliding_window_size):
        word_pairs = []
        for segment in segments:
            for i, word_start in enumerate(segment):
                for word_next in segment[i+1: i+sliding_window_size+1]:
                    pair = [word_start, word_next]
                    if pair not in word_pairs:
                        word_pairs.append(pair)
        return word_pairs
    
    def getAdjacencyMatrix(self, nodes : dict[(str, int)], edges : list[list[str]], accumulate = False, make_symmetric = False, norm = True):
        adjacency_matrix = np.zeros((len(nodes), len(nodes)))
        for word1, word2 in edges:
            i, j = nodes[word1], nodes[word2]
            if accumulate:
                adjacency_matrix[i,j] += 1
            else:
                adjacency_matrix[i,j] = 1

        if make_symmetric:
            adjacency_matrix = adjacency_matrix + adjacency_matrix.T + np.diag(adjacency_matrix.diagonal())
        
        if norm:
            column_wise_sum = np.sum(adjacency_matrix, axis=0)
            adjacency_matrix_normalised = np.divide(adjacency_matrix, column_wise_sum, where=column_wise_sum!=0)
            return adjacency_matrix_normalised
        return adjacency_matrix
                
                
    def analyze(self, text, pos_to_consider = ['NOUN', "PROPN", "VERB"], sliding_window_size = 4, lower = False, custom_stopwords = [], accumulate=True, make_symmetric=True, norm=True):
        '''
        ### text: str
        Text want to summarize and extract keywords.
        ### pos_to_consider: list[str]
        List of POS tags which will be considered. This must follow spaCy POS naming convention.
        ### sliding_window_size: int
        This will change the sliding window length which is being used in graph edge generation.
        ### lower: bool (True/False)
        If True the sentence will be lowerd before processing. Suggested to use True for better result.
        ### custom_stopwords: list[str]
        List of words which need to be skipped with the one being skipped in spaCy.
        ### accumulation: bool (True/False)
        If True the it will sum up the numberof incoming and outgoing connections from a node else will use binary if connected or not in the adjacency matrix of the graph.
        ### make_symmetric: bool (True/False)
        If True then the token graph will be undirected esle uni-directed.
        ### norm: bool (True/False)
        If True then the adjacency matrix will be normalized else not. Better to use True for more refined word.
        '''
        
        document = nlp(text)
        
        sentence_segments = self.getSentTokens(document, pos_to_consider, lower, custom_stopwords)
        
        vocab = self.createVocab(sentence_segments)
        
        word_pairs = self.createPairs(sentence_segments, sliding_window_size)
        
        graph_edges = self.getAdjacencyMatrix(vocab, word_pairs, accumulate=accumulate, make_symmetric=make_symmetric, norm=norm)
        
        epochs = self.epochs
        thresold = self.error_thresold
        damping = self.d
        token_weights = np.array([1]*len(vocab))
        previous_weights_sum = sum(token_weights)
        
        for i in range(epochs):
            token_weights = (1-damping) + damping * np.dot(graph_edges, token_weights)
            new_weights_sum = sum(token_weights)
            if abs(previous_weights_sum-new_weights_sum)<thresold:
                break
            previous_weights_sum = new_weights_sum
        effective_token_weights = []
        for word, i in vocab.items():
            effective_token_weights.append([word, token_weights[i]])
        
        self.token_weights = sorted(effective_token_weights, key=lambda x: x[1], reverse=True)
        
    def getKeywords(self, n_words = 10):
        '''### n_words: int
        Number of words need to be picked up.
        ### Returns: list[list[str, float]]
        List of list containing words and their calculated weight. 
        These weights are not uniform. It depend on only the document provided for analysis.'''
        return self.token_weights[:n_words]