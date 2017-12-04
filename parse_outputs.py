import pickle
with open('output_files.txt', 'r') as f:
	output_file_names = f.read().splitlines()



results = {}
for fname in output_file_names:
	with open(fname) as f2:

		result = f2.read().splitlines()[-1]

	results[fname] = result
	print(fname, 'result: ', result)


pickle.dump(results, open('results.p', 'wb'))








