export DATA_PATH=../data

# for TASK in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
for TASK in 13
do
	export VOCAB_SOURCE=${DATA_PATH}/vocab.txt
	export VOCAB_TARGET=${DATA_PATH}/vocab.txt
	export TRAIN_SOURCES=${DATA_PATH}/seq2seq/questions_${TASK}_train
	export TRAIN_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_train
	export DEV_SOURCES=${DATA_PATH}/seq2seq/questions_${TASK}_test
	export DEV_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_test
	export TEST_SOURCES=${DATA_PATH}/seq2seq/questions_${TASK}_test
	export TEST_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_test
	export MODEL_DIR=./models/tasks/task_${TASK}_fairseq

	sh conv_task.sh
done