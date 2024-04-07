import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model = joblib.load('model.pkl')

classifier = model['classifier']
cv = model['count_vectorizer']

genre_mapper = {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}

def input_processor(sample_script):
  sample_script = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_script)
  sample_script = sample_script.lower()
  sample_script_words = sample_script.split()
  sample_script_words = [word for word in sample_script_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_script = [ps.stem(word) for word in sample_script_words]
  final_script = ' '.join(final_script)
  return final_script

def genre_prediction(sample_script):
  temp = cv.transform([sample_script]).toarray()
  return classifier.predict(temp)[0]
