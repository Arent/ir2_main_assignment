export PYTHONIOENCODING=UTF-8

export DATA_PATH=../data/seq2seq

export VOCAB_SOURCE=${DATA_PATH}/vocab.txt
export VOCAB_TARGET=${DATA_PATH}/vocab.txt
export TRAIN_SOURCES=${DATA_PATH}/questions_1_train
export TRAIN_TARGETS=${DATA_PATH}/answers_1_train
export DEV_SOURCES=${DATA_PATH}/questions_1_test
export DEV_TARGETS=${DATA_PATH}/answers_1_test
export TEST_SOURCES=${DATA_PATH}/questions_1_test
export TEST_TARGETS=${DATA_PATH}/answers_1_test

export TRAIN_STEPS=1000

export MODEL_DIR=./models/qa_test

python -m bin.train \
  --config_paths="
      ./example_configs/qa_conv_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
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
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR