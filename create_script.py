from string import Template 
import os
import stat 

datasets =['en-10k']
learning_rates = ['0.001', '0.005', '0.01'] 
enc_types= [('bi', '1')]
# num_units = ['32', '64', '128', '256']
# embedding_sizes = ['32', '64', '128', '256']
dropouts = ['0.7']
cell_types = ['gru']
batch_sizes = ['8','16','32']
tasks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

taksk_per_model_type= {
	'attention':['2','3','5','6','7','8','14','16','17','18','19'],
	'normal':['2','3','5','7','9','10','12','13','14', '15', '16', '17', '18', '19', '20']
}

num_units_optimal = {
	'attention': {'1':256, '2':256, '3':256, '4':256, '5':256, '6':256, '7':256, '8':256, '9':256, '10':256, '11':256, '12':256, '13':256, '14':256, '15':256, '16':32, '17':128, '18':256, '19':64, '20':128},
	   'normal': {'1':256, '2':256, '3':256, '4':256, '5':64, '6':256, '7':128, '8':256, '9':256, '10':256, '11':256, '12':256, '13':64, '14':256, '15':256, '16':256, '17':256, '18':256, '19':128, '20':256},
}

embedding_sizes_optimal = {
	'attention': {'1':256, '2':256, '3':256, '4':256, '5':256, '6':256, '7':128, '8':128, '9':128, '10':256, '11':256, '12':256, '13':256, '14':128, '15':128, '16':256, '17':128, '18':256, '19':256, '20':32},
	'normal':{'1':256, '2':256, '3':256, '4':256, '5':64, '6':54, '7':256, '8':128, '9':128, '10':64, '11':256, '12':256, '13':32, '14':64, '15':256, '16':128, '17':256, '18':256, '19':256, '20':128},
}
	# 'sentence':['2','3','5','6','7','8','14','16','17','18','19'],

if not os.path.exists('grid_searches_lr_batch_10k_attention'):
    os.makedirs('grid_searches_lr_batch_10k_attention')
if not os.path.exists('grid_searches_lr_batch_10k_normal'):
    os.makedirs('grid_searches_lr_batch_10k_normal')    



for task in tasks:

  if not os.path.exists('grid_searches_lr_batch_10k_attention/task_%s' % task):
    os.makedirs('grid_searches_lr_batch_10k_attention/task_%s' % task)
    os.makedirs('grid_searches_lr_batch_10k_attention/task_%s/jobs' % task)
    os.makedirs('grid_searches_lr_batch_10k_attention/task_%s/outputs' % task)
    os.makedirs('grid_searches_lr_batch_10k_attention/task_%s/errors' % task)

  if not os.path.exists('grid_searches_lr_batch_10k_normal/task_%s' % task):
    os.makedirs('grid_searches_lr_batch_10k_normal/task_%s' % task)
    os.makedirs('grid_searches_lr_batch_10k_normal/task_%s/jobs' % task)
    os.makedirs('grid_searches_lr_batch_10k_normal/task_%s/outputs' % task)
    os.makedirs('grid_searches_lr_batch_10k_normal/task_%s/errors' % task)

job_file_template = Template('''#!/bin/bash
#SBATCH -N 1
#SBATCH -p $nodetype
#SBATCH -t $hours:$minutes:00
#SBATCH -o outputs/$name.output
#SBATCH -e errors/$name.error

module load python/3.5.2 gcc/5.2.0 cuda/8.0.44 cudnn/8.0-v6.0

python3 -u main.py --model_type=$model_type --data_dir=data/$data --cell_type=$cell_type --num_units=$num_unit --encoder_type=$encoder_type --vocab data/vocab.txt --model_dir models/dummy --task=$task --num_epochs=50 --dropout_keep_prob=$dropout --num_enc_layers=$num_enc_layers --embedding_size=$embedding_size --batch_size=$batch --learning_rate $lr
''')

name_template = Template('$model_type-$task-$data-batch=$batch-lr=$lr-type=$encoder_type-num_rnns=$encoder_units-h_size=$num_unit-emb=$embedding_size-dropout=$dropout-cell_type=$cell_type')


job_file_names = []






for model_type in ['normal', 'attention']:
	for task in taksk_per_model_type[model_type]:
		for dataset in datasets:
			for batch in batch_sizes:
				for lr in learning_rates:
					for enc_type in enc_types:
						for dropout in dropouts:
							for cell_type in cell_types:

								encoder_type, encoder_units= enc_type
								
								if task == "3":
									hours = '03'
									nodetype = "normal"
									minutes = '00'
								else:
									hours = '00'
									nodetype = "short"
									minutes = '50'

				
								num_unit = num_units_optimal[model_type][task]
								embedding_size = embedding_sizes_optimal[model_type][task]

								name = name_template.substitute(model_type=model_type, task=task, data=dataset, lr=lr, encoder_type=encoder_type,encoder_units=encoder_units, 
									num_unit=num_unit, embedding_size=embedding_size, dropout=dropout, cell_type=cell_type, batch=batch)

								job_file_names.append('grid_searches_lr_batch_10k_'+model_type+ '/task_'+task+'/jobs/'+ name)
								job_file_text = job_file_template.substitute(model_type=model_type, minutes=minutes, hours=hours, nodetype=nodetype, name=name, data=dataset, cell_type=cell_type, lr=lr,
																 num_unit=num_unit, encoder_type=encoder_type, dropout=dropout,
																  num_enc_layers=encoder_units, embedding_size=embedding_size, batch=batch, task=task)
								with open('grid_searches_lr_batch_10k_'+model_type+'/task_%s/jobs/' % task +name+'.job', 'w') as job_file:
									job_file.write(job_file_text)



with open('grid_searches_lr_batch_job_files' , 'w') as job_files_script:
	job_files_script.write('#!/bin/bash\n')
	for name in job_file_names:
		job_files_script.write("sbatch %s.job\n" % name)


st = os.stat('grid_searches_lr_batch_job_files' )
os.chmod('grid_searches_lr_batch_job_files', st.st_mode | stat.S_IEXEC)

