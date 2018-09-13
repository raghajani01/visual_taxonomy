from absl import flags
from absl import logging
from absl import app
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('scores', None, 'Path to the CSV file containing the model\'s scores.') 
flags.DEFINE_string('truths', None, 'Path to the CSV file containing ground-truth information (in 1-hot format).') 
flags.DEFINE_string('src', '', 'CSV of (1-based) indices of source categories.')
flags.DEFINE_string('dst', '', 'CSV of (1-based) indices of destination categories.')
flags.DEFINE_string('cnd', '', 'CSV of (1-based) indices of categories on which positive samples will be conditioned.')
flags.DEFINE_integer('num_labels', None, 'Number of labels (i.e. classes to expect.');

def ConvertCsvToInts(csv, num_labels):
   values = []
   for range_specs in list(filter(None, csv.split(','))):
       range_values = [int(i) for i in list(filter(None, range_specs.split(':')))]
       assert range_values, "@ConvertCsvToInts(): Range specs cannot be empty!"
       assert len(range_values) <= 2, "@ConvertCsvToInts(): Range specs cannot have more than 2 elements (it has {}!)".format(len(range_values))

       mn = range_values[0]
       mx = range_values[1] if len(range_values) == 2 else range_values[0]
       if (mn == -1):
           mn = 1
           mx = num_labels
       assert mx >= mn, "@ConvertCsvToInts(): mn (={}) > mx (={})!".format(mn, mx)
       for i in range(mn, mx+1):
           values.append(i)

   values = list(set(values))
   values.sort()

   return values

def GenerateConditionalExamples(scores, truths, conditional_set):
    with open(scores) as score_file, open(truths) as truth_file:
        for S, T in zip(score_file, truth_file):
            S = list(filter(None, S.split(',')))
            T = list(filter(None, T.split(',')))
            assert len(S) == len(T), "len(S)={} != len(T)={}".format(len(S), len(T))
        
            for i, t in enumerate(T):
                if float(t) == 1.0 and i+1 in conditional_set:
                    yield [float(s) for s in S]
                    break

def main(argv):
   assert FLAGS.scores is not None, "'--scores' has to be set."
   assert FLAGS.truths is not None, "'--truths' has to be set."
   assert FLAGS.num_labels is not None, "'--num_labels' has to be set."

   print('Reading predictions from "{}" ...'.format(FLAGS.scores))
   print('Reading ground-truths from "{}" ...'.format(FLAGS.truths))

   num_labels = int(FLAGS.num_labels)
   src = ConvertCsvToInts(FLAGS.src, num_labels)
   dst = ConvertCsvToInts(FLAGS.dst, num_labels)
   cnd = ConvertCsvToInts(FLAGS.cnd, num_labels)

   assert src, "'--src' has to be set."
   assert dst, "'--dst' has to be set."
   assert cnd, "'--cnd' has to be set."
   for class_label in src + dst + cnd:
       assert 1 <= class_label <= FLAGS.num_labels, "Class label ({}) out of range [1, {}].".format(class_label, FLAGS.num_labels)

   num_samples = 0
   all_scores = []
   for scores in GenerateConditionalExamples(FLAGS.scores, FLAGS.truths, cnd):
       num_samples = num_samples + 1
       all_scores.append(scores)

   print('Found {} relevant examples.'.format(num_samples))
   corr_mat = np.corrcoef(np.log(np.transpose(np.asarray(all_scores))))
   corr_mat = corr_mat[np.ix_([i - 1 for i in src], [i - 1 for i in dst])]
   print(corr_mat)

if __name__ == '__main__':
   app.run(main)
