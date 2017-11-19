import tensorflow as tf

class LSTM:
    def __init__(self, context, context_length, question, question_length, answer, 
        vocab_size, optimizer = tf.train.AdamOptimizer(1e-3), embedding_size_context=50, embedding_size_question=50, 
        hidden_layer_size=100):
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

            self.batch_size = tf.shape(context)[0]
            self.max_length = tf.shape(context)[1]

            #Set the optimizer
            self.optimizer = optimizer
            #Link to train, inference, loss and accuracy operations
            self._inference = None
            self._train = None
            self._loss = None
            self._accuracy = None

    @property
    def inference(self):
        #The first time this function is called the operations are made
        #Otherwise, return the inference operation
        if self._inference is None:
            #Set up the embeddings and lstm units for the context and question inputs
            with tf.variable_scope('context'):
                #Create embeddings for the context input
                embedding_matrix_context = tf.get_variable("embedding_matrix", \
                [self.vocab_size, self.embedding_size_context], dtype=tf.float32)
                embeddings_context = tf.nn.embedding_lookup(embedding_matrix_context, self.context) # [1, time, emb_size]

                #Create an lstm for the context
                lstm_context = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer_size)

                #Create the lstm embedded output for context
                encoder_outputs_context, encoder_state_context = tf.nn.dynamic_rnn(
                cell=lstm_context, inputs=embeddings_context, dtype=tf.float32,
                sequence_length=self.context_length, time_major=False)#, initial_state =initial_state_context)

            with tf.variable_scope("question"):
                #Create embeddings for the question input
                embedding_matrix_question = tf.get_variable("embedding_matrix", \
                        [self.vocab_size, self.embedding_size_question], dtype=tf.float32)
                embeddings_question = tf.nn.embedding_lookup(embedding_matrix_question, self.question) # [1, time, emb_size]

                lstm_question = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer_size)
                
                encoder_outputs_question, encoder_state_question = tf.nn.dynamic_rnn(
                cell=lstm_question, inputs=embeddings_question, dtype=tf.float32,
                sequence_length=self.question_length, time_major=False)#, initial_state =initial_state_context)


            #combine the lstm state of the  question and context
            combined_context_question = encoder_state_context[1] + encoder_state_question[1]
            
            #Use and MLP for the last layer
            prediction_weights = tf.get_variable("prediction_weights", [self.hidden_layer_size, self.vocab_size])
            
            self._inference = tf.matmul(combined_context_question, prediction_weights)
        

        return self._inference

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
            predicted_anser_ids = tf.argmax(self.inference, axis=1)
            self._accuracy = tf.reduce_mean(tf.cast(self.answer == predicted_anser_ids, tf.int32)) 

        return self._accuracy
