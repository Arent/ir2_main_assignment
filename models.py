import tensorflow as tf

class LSTM:
    def __init__(self, context, context_length, question, question_length, answer, 
        vocab_size, optimizer = tf.train.AdamOptimizer(1e-3), embedding_size_context=50, embedding_size_question=50, 
        hidden_layer_size=100, dropout=0.3, recurrent_cell =tf.nn.rnn_cell.BasicLSTMCell, context_cells =1, question_cells=1 ):
            ''' This function saves the placeholders and
            links to inference, train, loss and accuracy functions '''

            #Save the placeholders
            self.context = context
            self.context_length = context_length
            self.question = question
            self.question_length = question_length
            self.answer = answer

            #Set network dimensions
            self.vocab_size = vocab_size
            self.embedding_size_context= embedding_size_context
            self.embedding_size_question= embedding_size_question
            self.hidden_layer_size = hidden_layer_size

            self.recurrent_cell = recurrent_cell
            self.dropout = dropout
            self.context_cells = context_cells
            self.question_cells = question_cells

            #Set the optimizer
            self.optimizer = optimizer

            #Link to train, inference, loss and accuracy operations
            self._inference = None
            self._train = None
            self._loss = None
            self._accuracy = None
            self._prediction = None


    def _recurrent_block(self, scope_name, input, input_lengths, dropout=0, n_lstm=1, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):

            embedding_matrix = tf.get_variable("embedding_matrix", \
                [self.vocab_size, self.embedding_size_context], dtype=tf.float32)

            embeddings = tf.nn.embedding_lookup(embedding_matrix, input) # [1, time, emb_size]

            #Create an lstm cell
            lstm =[]
            for _ in range(n_lstm):
                cell = self.recurrent_cell(self.hidden_layer_size)
                cell = tf.contrib.rnn.DropoutWrapper(
                            cell, output_keep_prob=1.0 - dropout)
                lstm.append(cell)

            lstm = tf.contrib.rnn.MultiRNNCell(lstm)
            #Create the lstm embedded output 
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=lstm, inputs=embeddings, dtype=tf.float32,
                sequence_length=input_lengths, time_major=False)

            last_state = encoder_state[n_lstm-1]
            if isinstance(last_state, tuple):
                last_state = last_state[1]
            return encoder_state[n_lstm-1]
    
    @property
    def inference(self):
        #The first time this function is called the operations are made
        #Otherwise, return the inference operation
        if self._inference is None:
            #Set up the embeddings and lstm units for the context and question inputs
            encoder_state_context = self._recurrent_block('context', self.context, self.context_length, 
                dropout=self.dropout, n_lstm=self.context_cells)
            
            encoder_state_question = self._recurrent_block('question', self.question, self.question_length, 
                dropout=self.dropout, n_lstm=self.question_cells)
            
            #Use and MLP for the last layer
            with tf.variable_scope("prediction"):
                combined_context_question = encoder_state_context + encoder_state_question
                prediction_weights = tf.get_variable("weights", [self.hidden_layer_size, self.vocab_size])
                self._inference = tf.matmul(combined_context_question, prediction_weights)
        

        return self._inference

    @property
    def prediction(self):
        #The first time this function is called the operations are made
        #Otherwise, return the prediction operation
        #Similar to inference, but no dropout
        if self._prediction is None:
            #Set up the embeddings and lstm units for the context and question inputs
            encoder_state_context = self._recurrent_block('context', self.context, self.context_length, 
                dropout=0, n_lstm=self.context_cells, reuse=True)
            encoder_state_question = self._recurrent_block('question', self.question, self.question_length, 
                dropout=0,n_lstm=self.question_cells, reuse=True)
                        
            with tf.variable_scope("prediction", reuse=True):
                combined_context_question = encoder_state_context + encoder_state_question
                prediction_weights = tf.get_variable("weights", [self.hidden_layer_size, self.vocab_size])
                self._prediction = tf.matmul(combined_context_question, prediction_weights)
        

        return self._prediction
    @property
    def loss(self):
        if self._loss is None:
            labels=tf.one_hot(indices=self.answer, depth=self.vocab_size)
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.inference,labels=labels))

        return self._loss


    @property
    def train(self):
        if self._train is None:
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
            self._train = self.optimizer.apply_gradients(zip(gradients, variables))
        return self._train

    @property
    def accuracy(self): 
        if self._accuracy is None:            
            predicted_anser_ids = tf.cast(tf.argmax(self.inference, axis=1),tf.int32)
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self.answer, predicted_anser_ids), tf.float32)) 

        return self._accuracy
