export PYTHONIOENCODING=UTF-8

# parameters.yaml,
# --model_params "
python3 -m bin.train \
  --config_paths="
      ./example_configs/train_seq2seq.yml,
      ./example_configs/qa_metrics.yml" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --model "ConvSeq2Seq" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET
      attention.class: seq2seq.decoders.attention.AttentionLayerDot 
      attention.params:
        num_units: 32
      embedding.dim: $EMBED_SIZE
      encoder.class: seq2seq.encoders.ConvEncoderFairseq
      encoder.params:
        cnn.layers: $IN_LAYERS
        cnn.nhids: $IN_HIDDEN
        cnn.kwidths: $IN_KERNEL_SIZE 
        embedding_dropout_keep_prob: $KEEP_RATE
        nhid_dropout_keep_prob: $KEEP_RATE
      decoder.class: seq2seq.decoders.ConvDecoderFairseq
      decoder.params:
        cnn.layers: $OUT_LAYERS
        cnn.nhids: $OUT_HIDDEN
        cnn.kwidths: $OUT_KERNEL_SIZE
      optimizer.name: Adam
      optimizer.lr_decay_steps: $DECAY_STEPS
      optimizer.lr_decay_rate: $DECAY_RATE
      optimizer.params:
        epsilon: 0.0000008
      optimizer.learning_rate: $LEARNING_RATE
      source.max_seq_len: $MAX_LENGTH 
      source.reverse: false
      target.max_seq_len: 5
      position_embeddings.num_positions: $MAX_LENGTH 
" \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \ 
  --output_dir $MODEL_DIR \
  --eval_every_n_steps $MIN_EVAL \
  --save_checkpoints_steps $TRAIN_STEPS \
  --save_summary_steps $MIN_EVAL \
  --tf_random_seed 42
