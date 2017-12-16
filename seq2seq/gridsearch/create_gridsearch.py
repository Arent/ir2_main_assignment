import itertools

parameters = {}
parameters['task'] = [9]
parameters['decay_rate'] = [0.9, 1.0]
parameters['decay_step'] = [1000, 5000]
parameters['learning_rate'] = [0.01, 0.001]
parameters['keep_rate'] = [0.9]
parameters['embed_sizes'] = [16,32] #num_h, kernel_size
parameters['batch_sizes'] = [50]
parameters['num_layers'] = [2,4,6]

id2key = parameters.keys()
grid_params = list(itertools.product(*[parameters[key] for key in id2key]))

for params in grid_params:
    search = {key: param for (key, param) in zip(id2key, params)}
    search['hidden'] = ",".join([str(search['embed_sizes']) for _ in range(search['num_layers'])])
    
    file_name = "SEARCH_" + ("-".join([key + "_" + str(search[key]) for key in id2key])).replace(".", "_")

    header = """#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH -t 00:30:00
#SBATCH -o {name}.output
#SBATCH -e {name}.error

module load python/3.5.2 gcc/5.2.0 cuda/8.0.44 cudnn/8.0-v6.0

export MODEL_DIR=./models/tasks/{name}
export DATA_PATH=../data""".format(name=file_name)

    content = """
export DATA_PATH=../data

export TRAIN_STEPS=15000
export MIN_EVAL=1000
export BATCH_SIZE=50
export LEARNING_RATE={learning_rate}
export SAVE_CHECKPOINTS_STEPS=$TRAIN_STEPS
export KEEP_RATE={keep_rate}
export EMBED_SIZE={embed_sizes}
export IN_LAYERS={num_layers}
export IN_HIDDEN={hidden}
export IN_KERNEL_SIZE={hidden}
export OUT_LAYERS=2
export OUT_HIDDEN=16,16
export OUT_KERNEL_SIZE=4,4
export MAX_LENGHT=1200 #1891
export DECAY_STEPS={decay_step}
export DECAY_RATE={decay_rate}
export TASK={task}
    """.format(**search)

    run = """
export VOCAB_SOURCE=${DATA_PATH}/vocab.txt
export VOCAB_TARGET=${DATA_PATH}/vocab.txt
export TRAIN_SOURCES=${DATA_PATH}/seq2seq/questions_context_${TASK}_train
export TRAIN_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_train
export DEV_SOURCES=${DATA_PATH}/seq2seq/questions_context_${TASK}_val
export DEV_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_val
export TEST_SOURCES=${DATA_PATH}/seq2seq/questions_context_${TASK}_test
export TEST_TARGETS=${DATA_PATH}/seq2seq/answers_${TASK}_test

sh conv_task.sh
    """
    job_content = header + content + run

    with open(file_name + ".job", 'w') as f:
         f.write(job_content)

