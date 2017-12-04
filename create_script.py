from string import Template 
import os
import stat 

datasets =['en']
learning_rates = ['0.0002', '0.0005', '0.001'] 
enc_types= [('bi', '1'), ('uni', '1'), ('uni', '2')]
num_units = ['16', '32', '64', '128', '256']
embedding_sizes = ['20', '50', '100']
dropouts = ['1.0', '0.8', '0.6', '0.4']
cell_types = ['gru']
batch_sizes = ['10', '50']


if not os.path.exists('jobs'):
    os.makedirs('jobs')

if not os.path.exists('outputs'):
    os.makedirs('outputs')

if not os.path.exists('errors'):
    os.makedirs('errors')



job_file_template = Template('''#!/bin/bash
#SBATCH -N 1
#SBATCH -p short
#SBATCH -t 00:$minutes:00
#SBATCH -o outputs/$name.output
#SBATCH -e errors/$name.error

module load python/3.5.2 gcc/5.2.0 cuda/8.0.44 cudnn/8.0-v6.0

python3 -u main.py --debug=True --use_attention=True --data_dir=data/$data --cell_type=$cell_type --num_units=$num_unit --encoder_type=$encoder_type --task=1 --num_epochs=20 --dropout_keep_prob=$dropout --num_enc_layers=$num_enc_layers --embedding_size=$embedding_size --batch_size=$batch
''')

name_template = Template('$data-lr=$lr-type=$encoder_type-num_rnns=$encoder_units-h_size=$num_unit-emb=$embedding_size-dropout=$dropout-cell_type=$cell_type')


job_file_names = []


for dataset in datasets:
	for batch in batch_sizes:
		for lr in learning_rates:
			for enc_type in enc_types:
				for num_unit in num_units:
					for embedding_size in embedding_sizes:
						for dropout in dropouts:
							for cell_type in cell_types:

								encoder_type, encoder_units= enc_type
								if int(embedding_size) > 128 or int(num_unit) > 128:
									minutes = '40'
								else:
									minutes = '20'

								name = name_template.substitute(data=dataset, lr=lr, encoder_type=encoder_type,encoder_units=encoder_units, 
									num_unit=num_unit, embedding_size=embedding_size, dropout=dropout, cell_type=cell_type)

								job_file_names.append(name)
								job_file_text = job_file_template.substitute(minutes=minutes,name=name, data=dataset, cell_type=cell_type, 
																 num_unit=num_unit, encoder_type=encoder_type, dropout=dropout,
																  num_enc_layers=encoder_units, embedding_size=embedding_size, batch=batch)

								with open('jobs/'+name+'.job', 'w') as job_file:
									job_file.write(job_file_text)



with open('job_files', 'w') as job_files_script:
	job_files_script.write('#!/bin/bash\n')
	for name in job_file_names:
		job_files_script.write("sbatch jobs/%s.job\n" % name)


st = os.stat('job_files')
os.chmod('job_files', st.st_mode | stat.S_IEXEC)


with open('output_files.txt', 'w') as output_file:
	for name in job_file_names:
		output_file.write("outputs/%s.output\n" % name)


