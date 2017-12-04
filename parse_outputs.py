with open('output_files.txt', 'r') as f:
	output_file_names = f.readlines()



results = {}
for fname in output_file_names:
	with open(fname) as f2:
		result = f2.readlines()[-1]

	results[fname] = result
	print(fname, 'result: ', result)


pickle.dump(results, open('results.p', 'w'))








