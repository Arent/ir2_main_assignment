import itertools

parameters = { }
parameters['task'] = [19]
parameters['learning_rate'] = [ 0.001, 0.0001]
parameters['keep_rate'] = [1.0, 0.8]
parameters['embed_sizes'] = [16, 32, 64]  # num_h, kernel_size
parameters['batch_sizes'] = [32, 64, 128]
parameters['num_layers'] = [2, 4, 6]
parameters['kernel_sizes'] = [16]
parameters['max_epochs'] = [400]

id2key = parameters.keys()
grid_params = list(itertools.product(*[parameters[key] for key in id2key]))

for params in grid_params:
    search = { key: param for (key, param) in zip(id2key, params) }
    search['file_name'] = "SEARCH_" + "-".join([key + "_" + str(search[key]) for key in id2key])

    header = """#!/bin/bash
    #SBATCH -N 1
    #SBATCH -p gpu_short
    #SBATCH -t 00:15:00
    #SBATCH -o {name}.output
    #SBATCH -e {name}.error

    module load python/3.5.2 gcc/5.2.0 cuda/8.0.44 cudnn/8.0-v6.0
    
    export MODEL_DIR=./models/tasks/{name}
    export DATA_PATH=../data""".format(name=search['file_name'])

    job_content = """#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 00:05:00
#SBATCH -o cat_logs/{name}.output
#SBATCH -e cat_logs/{name}.error

module load python/3.5.2 gcc/5.2.0 cuda/8.0.44 cudnn/8.0-v6.0
""".format(name=search['file_name']) + """
srun -u python3 cnnde_train.py --data_dir=tasks/en/ --model_dir=logs/ --task {task} --vocab=data/vocab.txt --batch_size {batch_sizes} --learning_rate {learning_rate} --dropout_keep_prob {keep_rate} --num_epochs {max_epochs} --num_layers {num_layers} --embedding_size {embed_sizes} --kernel_size 16
    """.format(**search)


    with open(search['file_name'] + ".job", 'w') as f:
        f.write(job_content)
