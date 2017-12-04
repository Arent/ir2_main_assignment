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
      embedding.dim: 32
      encoder.class: seq2seq.encoders.ConvEncoderFairseq
      encoder.params:
        cnn.layers: 2
        cnn.nhids: 32,32
        cnn.kwidths: 10,10
        embedding_dropout_keep_prob: 0.9
        nhid_dropout_keep_prob: 0.9
      decoder.class: seq2seq.decoders.ConvDecoderFairseq
      decoder.params:
        cnn.layers: 1
        cnn.nhids: 32
        cnn.kwidths: 4
      optimizer.name: Adam
      optimizer.params:
        epsilon: 0.0000008
      optimizer.learning_rate: 0.001
      source.max_seq_len: 70
      source.reverse: false
      target.max_seq_len: 2
" \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR \
  --eval_every_n_steps $MIN_EVAL \
  --save_checkpoints_steps $TRAIN_STEPS \
  --save_summary_steps $MIN_EVAL \
  --tf_random_seed 42