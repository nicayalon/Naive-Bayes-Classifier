import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import math

# TODO: Change data path to reflect your own file structure! 
# Or ensure train.csv and test.csv are in the same directory from which you will run this file.
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def preprocess(text):
    stop_words = [' ', '', 'this', 'the', 'a', 'an', 'and', 'are', 'also', 'by', 'of', 'not', 'however', 'in', 'some', 'is', 'as', 'can', 'those', 'we', 'to', 'it', 'has', 'on', 'be', 'there', 'been', 'for', 'that', 'how', 'which', 'then']
    new_text = re.sub('[^A-Za-z]', ' ', text)
    new_text = new_text.lower()
    text_as_list = new_text.split(' ')
    text_as_list = [x for x in text_as_list if x not in stop_words]
    new_text = ' '.join(text_as_list)
    return new_text

class BayesClassifier:
    
    def __init__(self):
        self.class_occurence = { 'astro-ph': 0, 'astro-ph.CO': 0, 'astro-ph.GA': 0, 'astro-ph.SR': 0, 'cond-mat.mes-hall': 0, 'cond-mat.mtrl-sci': 0, 'cs.LG': 0, 'gr-qc': 0, 'hep-ph': 0 , 'hep-th': 0, 'math.AP': 0, 'math.CO': 0, 'physics.optics': 0, 'quant-ph': 0, 'stat.ML': 0} 
        self.vocab_counter_by_class = { 'astro-ph': {}, 'astro-ph.CO': {}, 'astro-ph.GA': {}, 'astro-ph.SR': {}, 'cond-mat.mes-hall': {}, 'cond-mat.mtrl-sci': {}, 'cs.LG': {}, 'gr-qc': {}, 'hep-ph': {} , 'hep-th': {}, 'math.AP': {}, 'math.CO': {}, 'physics.optics': {}, 'quant-ph': {}, 'stat.ML': {}} # Format { 0: {'invisible': 2, 'decay': 3, 'standard': 1, 'model': 1 ...}, 1: {'conclude': 1, 'that': 3, 'area': 2 ...}}
        self.number_examples_seen = 0
        self.vocab = {}

    def build_vocab(self, train_data):
        i = 0
        for line in train_data['Abstract']:
            self.number_examples_seen += 1 # Keep track of total number of training examples seen
            curr_class = train_data['Category'][i] # Used to keep track of word occurences by class
            if curr_class in self.class_occurence:
                self.class_occurence[curr_class] += 1
            else:
                self.class_occurence[curr_class] = 1
            line = preprocess(line)
            for word in line.split(' '):
                if word in self.vocab_counter_by_class.get(curr_class):
                    self.vocab_counter_by_class[curr_class][word] += 1 
                else:
                    self.vocab_counter_by_class[curr_class][word] = 1
                if word not in self.vocab: # Keeps track of the total vocabulary of all documents
                    self.vocab[word] = 1
            i= i+1

    def predict_class(self, abstract):
        all_predictions = { 'astro-ph': 0, 'astro-ph.CO': 0, 'astro-ph.GA': 0, 'astro-ph.SR': 0, 'cond-mat.mes-hall': 0, 'cond-mat.mtrl-sci': 0, 'cs.LG': 0, 'gr-qc': 0, 'hep-ph': 0 , 'hep-th': 0, 'math.AP': 0, 'math.CO': 0, 'physics.optics': 0, 'quant-ph': 0, 'stat.ML': 0} # {'class1' : chance of being in class 1, etc}
        curr_vocab = {}
        abstract = preprocess(abstract)
        for word in abstract.split(' '):
            if word in curr_vocab: # Build a vocabulary of the words in the abstract of the current example
                curr_vocab[word] += 1 
            else:
                curr_vocab[word] = 1

        # Calculate P(c|d) = P(d|c)*P(c)
        for c in all_predictions:
            prob = 0
            curr_class_vocab_word_count = self.vocab_counter_by_class[c]
            for word in curr_vocab:
                if word in curr_class_vocab_word_count:
                    prob += curr_vocab.get(word,1)*math.log(curr_class_vocab_word_count[word] + 1)

            all_predictions[c] = prob + math.log(self.class_occurence[c] / self.number_examples_seen)

        return max(all_predictions, key=all_predictions.get)

B = BayesClassifier()

# ----- TRAINING ------- 

B.build_vocab(train_data)

# ----- PREDICTING ------- 

# Make a new matrix to store test outputs
results = np.empty([len(test_data),1], dtype="<U20")
i = 0
for line in test_data.head(10)['Abstract']:
    # print(i, end = ' ')
    results[i] = B.predict_class(line)
    i = i+1

pd.DataFrame(results, columns=['Category']).to_csv("output.csv")


