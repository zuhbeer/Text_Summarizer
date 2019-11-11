import numpy as np
import pandas as pd 
import requests, re 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import warnings
from attention import AttentionLayer
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
from nltk.translate.bleu_score import sentence_bleu

#this script will train on any csv with a 'Text' and 'Summary' column
#read in processed training text data

data = pd.read_csv('training.csv')

def clipper(data, i=100, j=10):

    #set wordcount threshold for training
    #It is difficult for the encoder to memorize long sequences into a fixed length vector
    max_text_len=i
    max_summary_len=j

    cleaned_text =np.array(data['Text'])
    cleaned_summary=np.array(data['Summary'])

    short_text=[]
    short_summary=[]

    for i in range(len(cleaned_text)):
        if(len(cleaned_summ[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summ[i])

    #initialize df with clipped data
            
    df=pd.DataFrame({'text':short_text,'summary':short_summary})

    #adding start of special token and end of special token to summaries
    #these are markers to start and end encoding/decoding process

    df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')

    return df

def split_process(df):
#train on 90% of the data, shuffle true

    from sklearn.model_selection import train_test_split
    x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),
                                        np.array(df['summary']),
                                        test_size=0.1,random_state=0,
                                        shuffle=True)

    #tokenizer builds the vocab and converts word sequence to an integer sequence

    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_tr))

    #convert text sequences into integer sequences
    x_tr    =   x_tokenizer.texts_to_sequences(x_tr) 
    x_val   =   x_tokenizer.texts_to_sequences(x_val)

    #Pads sequences to the same length
    #padding zero at end of sequence up to maximum length
    x_tr    =   pad_sequences(x_tr,  maxlen=max_len_text, padding='post') 
    x_val   =   pad_sequences(x_val, maxlen=max_len_text, padding='post')

    x_voc = len(x_tokenizer.word_index) +1

    #same for summary

    #preparing a tokenizer for summary on training data 
    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_tr))

    #convert summary sequences into integer sequences
    y_tr    =   y_tokenizer.texts_to_sequences(y_tr) 
    y_val   =   y_tokenizer.texts_to_sequences(y_val) 

    #padding zero upto maximum length
    y_tr    =   pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
    y_val   =   pad_sequences(y_val, maxlen=max_len_summary, padding='post')

    y_voc = len(y_tokenizer.word_index) +1

    return x_tr,x_val,y_tr,y_val,x_voc,y_voc

def Model_building(x_voc, y_voc, ld = 300, ed = 100):
    
    ''' three LSTM layer RNN '''

    from keras import backend as K 
    K.clear_session()

    # Latent dimensionality of the encoding space
    #input_dim: Size of the vocabulary
    #output_dim: embedding_dim, Dimension of the dense embedding
    latent_dim = ld
    embedding_dim=ed

    # Encoder, instantiate a Keras tensor of specified shape
    encoder_inputs = Input(shape=(max_text_len,))

    '''#embedding layer turns positive integers (indexes) into dense vectors of fixed size, takes input and output size
    ##trainable allows changing of weights during training
    # output 3D tensor with shape: batch_size, input_length, output_dim'''
    enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

    '''LSTM layer will choose different implementations to maximize the performance
    #When the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep
    #When return state = True, LSTM produces the hidden state and cell state of the last timestep only
    #dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
    #recurrent_dropout: Fraction of the units to drop for the linear transformation of the recurrent state.
    #encoder LSTM turns input sequences to 2 state vectors'''

    #encoder lstm 1:
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    #encoder lstm 3
    encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))

    #embedding layer
    dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    '''attention layer: attention mechanism equips a neural network with 
    the ability to focus on a subset of its inputs.
    instead of looking at all the words in the source sequence, 
    we can increase the importance of specific parts 
    of the source sequence that result in the target sequence:
    using bahdanau attention '''
    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # Concat attention input and decoder LSTM output
    #returns a single tensor from a list of input tensors
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    #dense layer
    # time dist. wrapper allows to apply a layer to every temporal slice of an input
    decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    print(model.summary())
    return model

def compiler(x_tr,x_val,y_tr,y_val,
            learning_rate = 0.0025, 
            lr_decay = 0.00001, 
            ep=30, 
            pat=3
            bat=128):


    #keras recomends rmsprop, but adam incorporates the heuristics of rmsprop and momentum
    opt = Adam(lr=learning_rate,decay=lr_decay)

    #sparse_cat_crossent converts integer sequence to a one-hot vector on the fly
    #saves memory, faster that cat_crossent
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')

    #stop training early if model does not improve performance after 4 epochs
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=pat)

    #keras recomends atleast 20 epochs for coherant english
    history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],
                    y_tr.shape[1], 1)[:,1:] ,epochs=ep,callbacks=[es],
                    batch_size=bat, validation_data=([x_val,y_val[:,:-1]], 
                    y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

#build the dictionary to convert the index to word for target and source vocabulary
#Reverse-lookup token index to decode sequences back to something readable.
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

#Inference 

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])



#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

#implementation of the inference process 

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

#define the functions to convert an integer sequence to a word sequence for summary as well as the reviews
def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


model.save('model.h5')
encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')

summ_df = pd.DataFrame()

def summary_df(entries = 10000):
    rev = []
    tru = []
    summ = []
    for i in range(entries):
        rev.append(seq2text(x_tr[i]))
        tru.append(seq2summary(y_tr[i]))
        summ.append(decode_sequence(x_tr[i].reshape(1,max_text_len)))
    summ_df['reviews'] = rev
    summ_df['true'] = tru
    summ_df['predicted'] = summ
    summ_df.to_csv('Summarized')
    return tru, summ


def bleu_score(tru, summ):
    '''minor cleaning and formatting 
    to feed into BLEU scorer'''
    tru = [x.strip(' ') for x in tru]
    summ = [x.strip(' ') for x in summ]
    actual = []
    summary = []
    for i in range(len(tru)):
        actual.append(tru[i].split(' '))
        summary.append(summ[i].split(' '))
    
    oneg = []
    twog = []
    threeg = []
    fourg = []
    for i in range(len(actual)):
        reference = [actual[i]]
        candidate = summary[i]
        oneg.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        twog.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        threeg.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        fourg.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
    
    one_gram = np.asarray(oneg)
    two_gram = np.asarray(twog)
    three_gram = np.asarray(threeg)
    four_gram = np.asarray(fourg)

    #It is common to report the cumulative BLEU-1 to BLEU-4 scores 
    # when describing the skill of a text generation system
    print("One Gram Score: ", np.mean(one_gram))
    print("Four Gram Score: ", np.mean(four_gram)))