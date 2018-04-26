from keras.preprocessing.sequence import pad_sequences

def pad_idx_seqs(idx_seqs, max_seq_len, value=0.0):
    # Keras provides a convenient padding function; 
    padded_idxs = pad_sequences(sequences=idx_seqs, maxlen=max_seq_len, 
                                value=value)
    return padded_idxs

def get_max_seq_len(sents):
    return max([len(idx_seq) for idx_seq in sents])