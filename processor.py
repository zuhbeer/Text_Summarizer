# IMPORTs
import numpy as np
import pandas as pd

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import ShuffleSplit
#from keras import model
from sklearn.metrics import accuracy_score
import pickle #rick
import itertools
# stats mean
# multilabel binarizer
# nltk bleu scores




class DataProcessor(object):
    def __init__(self,file_name, hold_out_file, sample_ratio = None):
        self.file_name = file_name
        self.data_df = pd.read_csv(self.file_name,dtype=str)
        
        if hold_out_file:
            self.hold_out_file = hold_out_file
            self.hold_out = pd.read_csv(self.hold_out_file,dtype=str)
            
        print('columns')
        print(self.data_df.columns)
        
        print(pd.__version__)
        
        self.data_df.loc[:,self.data_df.columns[1]] = self.process_start_tokens(self.data_df.loc[:,self.data_df.columns[1]])
        
        if sample_ratio:
            self.data_df = self.data_df.sample(frac=sample_ratio, replace = True, random_state=1)
            self.hold_out = self.hold_out.sample(frac=sample_ratio,replace=True,random_state=1)
        
        #target
        self.data_df['target'] = self.data_df.loc[:,self.data_df.columns[1]].apply(lambda x: str(x)[1:])
        
        
        # spanish source  ^bano$
        # english target   ^bathroom$
        # algo target       bathroom$
        
        self.source_seq = self.data_df.loc[:,self.data_df.columns[0]].tolist()
        self.intermediate_seq = self.data_df.loc[:,self.data_df.columns[1]].tolist()
        self.target_seq = self.data_df.target.tolist()
        
        rs = ShuffleSplit(n_splits=2, test_size=.2, random_state = 24)
        
        self.train_idx = list(rs.split(self.data_df.index))[0][0]
        self.test_idx = list(rs.split(self.data_df.index))[0][1]
        
        
    def set_attributes(self):
        self.temp_df = pd.read_csv(self.file_name, dtype=str).drop_duplicates()
        #if self.hold_out_file:
        #   self.temp_df2 = pd.read_csv(self.hold_out_file, dtype=str).drop_duplicates()
        #    self.temp_df = pd.concat([self.temp_df,self.temp_df2])
            
        self.temp_df.loc[:,self.temp_df.columns[1]] = self.process_start_tokens(self.temp_df.loc[:,self.temp_df.columns[1]])
        self.temp_source = self.temp_df.loc[:,self.temp_df.columns[0]].tolist()
        self.temp_target = self.temp_df.loc[:,self.temp_df.columns[1]].tolist()
        
        
        # artifacts
        
        #self.tokenizer = self.create_tokenizer(self.temp_source + self.temp_target, char_level = True)
        #self.vocab_size = len(self.tokenizer.word_index) + 1
        #self.source_length = self.max_length(self.temp_source, char_level=True)
        #self.target_length = self.max_length(self.temp_target, char_level=True)
        
        #print(' max source length : %d' % (self.source_length))
        #print(' max target length : %d' % (self.target_length))
        #print(' vocab size: %d' % (self.vocab_size))
        
        
    def process_start_tokens(self,series):
        series.apply(lambda x: '^' + str(x) + '$')
        return series
    
    
#     def create_tokenizer(self, lines, char_level=True):
#         if char_level:
#             tokenizer=Tokenizer(split = ' ', oov_token='#', char_level=True)
#             print('character level tokenizer used')
            
#         else:
#             tokenizer= Tokenizer(filters = '!"#%&()*+,-', oov_token='   ')
#             print('word level tokenizer used')
            
#         tokenizer.fit_on_texts(lines)
#         return tokenizer
        
  

     #def max_length(self, lines, char_level=True):
#         if char_level:
#             return max(len(line) for line in lines)
#         else:
#             return max(len(line.split()) for line in lines)
        
        
        
    def encode_data(self,series, tokenizer, max_len, vocab, rank):
        
        data = series.tolist()
        encoded_data = tokenizer.texts_to_sequences(data)
        
        # padding
        encoded_data = pad_sequences(encoded_data, maxlen=max_len, padding='post')
        
        
        if rank > 2:
            nparrayshape = (len(data),max_len, vocab)
            hot_encoded_data = np.zeros(dtype='float32',shape=nparrayshape)
            
            index = 0
            
            for element in encoded_data:
                hot_encoded_element = to_categorical(element, num_classes=vocab)
                hot_encoded_data[index] = hot_encoded_element
                index += 1
                
            del data, encoded_data, nparrayshape, index
            return hot_encoded_data
        
        
        else:
            del data
            return encoded_data
        
    def data_generator(self,dataset,batch_size=64):
        dataset = dataset[:(len(dataset)-(len(dataset) % batch_size))]
        total_len = len(dataset)
        
        iterations = 0
        start_idx = 0
        
        
        while True:
            if start_idx==total_len:
                start_idx = 0
            
            current_data = dataset[start_idx:start_idx+batch_size]
            
            start_idx +=batch_size
            iterations += 1
            
            source_batch = self.encode_data(current_data.iloc[:,0]
                                              ,self.source_tokenizer
                                              ,self.source_length
                                              ,self.vocab_size
                                              ,rank=0)
            
                        
            intermediate_batch = self.encode_data(current_data.iloc[:,1]
                                              ,self.target_tokenizer
                                              ,self.target_length
                                              ,self.vocab_size
                                              ,rank=0)
                        
            target_batch = self.encode_data(current_data.iloc[:,2]
                                              ,self.target_tokenizer
                                              ,self.target_length
                                              ,self.vocab_size
                                              ,rank=3)
            
            
            yield([source_batch,intermediate_batch],target_batch)
            
            
            
            
      
            
            
        
        
        
        
        
        
        
        
        
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
              
            
            
            
            
            
            
            
            
            
            