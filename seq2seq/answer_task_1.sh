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

export MODEL_DIR=./models/qa_test

export PRED_DIR=./predictions
mkdir -p ${PRED_DIR}

python3 -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt