export DATA_PATH=../data

export TRAIN_STEPS=10000
export MIN_EVAL=1000
export SAVE_CHECKPOINTS_STEPS=TRAIN_STEPS

# for TASK in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# for TASK in "1-10k" "2-10k" "3-10k" "4-10k" "5-10k" "6-10k" "7-10k" "8-10k" "9-10k" "10-10k" "11-10k" "12-10k" "13-10k" "14-10k" "15-10k" "16-10k" "17-10k" "18-10k" "19-10k" "20-10k"
# for TASK in 5
for TASK in "1-10k" 
do
	export VOCAB_SOURCE=${DATA_PATH}/vocab.txt
	export VOCAB_TARGET=${DATA_PATH}/vocab.txt
	export TRAIN_SOURCES=${DATA_PATH}/seq2seq/questions_${TASK}_train
	export TRAIN_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_train
	export DEV_SOURCES=${DATA_PATH}/seq2seq/questions_${TASK}_test
	export DEV_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_test
	export TEST_SOURCES=${DATA_PATH}/seq2seq/questions_${TASK}_test
	export TEST_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_test
	export MODEL_DIR=./models/tasks/task_${TASK}_fairseq_attention

	sh conv_task.sh
done