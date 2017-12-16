import pickle
import glob

#with open('output_files.txt', 'r') as f:
#  output_file_names = f.read().splitlines()

output_file_names = glob.glob("outputs/*.output")

num_results = 0
best_result = 0.0
best_result_setting = None
passes = []
results = {}
for fname in output_file_names:

  try:
    with open(fname) as f2:
      result = f2.read().splitlines()
      if len(result) <= 1:
        print("W: Skipping", fname)
        continue
      num_epochs_needed = result[-2]
      result = result[-1]

    results[fname] = result
    try:
      cur_result = float(result)
      num_results += 1
      if best_result_setting is None or cur_result > best_result:
        best_result_setting = fname
        best_result = cur_result
      if cur_result >= 0.0:
        num_epochs_needed = 40 #if num_epochs_needed == "None" else num_epochs_needed
        passes.append((fname, cur_result, int(num_epochs_needed)))
        #print(fname, cur_result, num_epochs_needed)
      #print(fname, 'result: ', float(result))
    except ValueError:
      print("W: Skipping ", fname)
      continue
  except IOError:
    print("W: Skipping ", fname)
    continue

#print(best_result_setting)
#print(best_result)

sorted_passes = sorted(passes, key=lambda t: (t[1], -t[2]))
for passed in sorted_passes:
  print(passed)

print("%d outputs parsed." % num_results)
print("%d settings passed." % len(passes))

#pickle.dump(results, open('results.p', 'wb'))
