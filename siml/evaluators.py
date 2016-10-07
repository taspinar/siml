
def true_positives(determined_Y, real_Y):
  true_positives = 0
  for ii in range(0,len(determined_Y)):
    if determined_Y[ii] == 1 and real_Y[ii] == 1: 
      true_positives+=1
  return true_positives

def all_positives(determined_Y):
  return sum(determined_Y)

def false_negatives(determined_Y, real_Y):
  false_negatives = 0
  for ii in range(0,len(determined_Y)):
    if determined_Y[ii] == 0 and real_Y[ii] == 1: 
      false_negatives+=1
  return false_negatives
  
def precision(determined_Y, real_Y):
    return true_positives(determined_Y, real_Y) / float(all_positives(determined_Y))

def recall(determined_Y, real_Y):
    return true_positives(determined_Y, real_Y) / float((true_positives(determined_Y, real_Y) + false_negatives(determined_Y, real_Y)))

def f1_score(determined_Y, real_Y):
    p = precision(determined_Y, real_Y)
    r = recall(determined_Y, real_Y)
    f1 = 2 * (p * r) / (p + r)
    return f1
