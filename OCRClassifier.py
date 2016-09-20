'''
Lorin Vandegrift
OCRClassifier classifies 128 bit binary le
'''

import random
import numpy as np
import matplotlib.pyplot as plt

class OCR_classifier:

	def __init__(self):
		self.sample_data = []
		self.weight_vector = []
		self.test_data = []		
		self.bias = 0
		self.c_weight_vector = []
		self.c_bias = 0

	def read_sample_data(self, file_path):
		try:
			with open(file_path, 'r') as workfile:	
				for line in workfile:
					line_parts = line.split('\t')				
					if len(line_parts[0]) > 1:						
						descriptor = [int(i) for i in line_parts[1][2:]]				
						result_class = line_parts[2]					
						self.sample_data.append({'descriptor':descriptor, 'result_class':result_class})				
				return True
		except:
			return False

	def read_test_data(self, file_path):
		try:
			with open(file_path, 'r') as workfile:	
				for line in workfile:
					line_parts = line.split('\t')				
					if len(line_parts[0]) > 1:						
						descriptor = [int(i) for i in line_parts[1][2:]]				
						result_class = line_parts[2]					
						self.test_data.append({'descriptor':descriptor, 'result_class':result_class})				
				return True
		except:
			return False

	def generate_learning_rate(self, y, item, score, l_type = "PA"):
		return 1		
		if l_type == "PA":			
			return (1 - y * score) / sum([i ** 2 for i in item['descriptor']])
		else:			
			return 1

	def generaate_multi_class_learning_rate(self, l_type = "PA"):
		if l_type == "PA":
			return (1)

	def setup_binary_weights(self):
		if len(self.sample_data) == 0:
			return False
		self.weight_vector = [0] * len(self.sample_data[0]['descriptor'])		
		self.c_weight_vector = [0] * len(self.sample_data[0]['descriptor'])		
		self.bias = 0

	def setup_multi_class_weights(self):
		if len(self.sample_data) == 0:
			return False
		self.weight_vector = [0] * len(self.sample_data[0]['descriptor']) * 26				
		self.bias = 0

	def train_binary_classifier(self, iter_num):
		positive = ['a','e','i','o','u']		
		if len(self.sample_data) == 0 or len(self.weight_vector) == 0:
			return -1				
		for i in range(iter_num):
			count = 0
			random.shuffle(self.sample_data)
			for item in self.sample_data:				
				score = float(np.dot(self.weight_vector, item['descriptor'])) + self.bias
				
				if item['result_class'] in positive:
					y = 1
				else:
					y = -1
				
				if score * y <= 0:
					
					learning_rate = self.generate_learning_rate(y, item, score)					
					self.c_weight_vector += np.array(item['descriptor']) * y * learning_rate * count
					self.c_bias += (y * count)
					count = 0
					self.bias += y					
					self.weight_vector += np.array(item['descriptor']) * y * learning_rate  #/ sum([i ** 2 for i in point[:-1]])
				else:
					count += 1	

		self.weight_vector -= (1/iter_num)*self.c_weight_vector
		self.bias -= (1/iter_num)*self.c_bias
			

	def test_binary_classifier(self):
		positive = ['a','e','i','o','u']
		if len(self.test_data) == 0 or len(self.weight_vector) == 0:
			return -1	
		num_mistakes = 0	
		for item in self.test_data:				
			score = np.dot(self.weight_vector, item['descriptor']) + self.bias
			if item['result_class'] in positive:
				y = 1
			else:
				y = -1

			if score * y <= 0:				
				num_mistakes += 1
				

		return num_mistakes

	def train_multi_class_classifier(self, iter_num):
		letters = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}
		if len(self.sample_data) == 0 or len(self.weight_vector) == 0:
			return False
		for i in range(iter_num):
			count = 0
			random.shuffle(self.sample_data)
			for item in self.sample_data:
				results = []
				for j in range(len(letters)):					
					results.append(np.dot(item['descriptor'], self.weight_vector[j*128:j*128 + 128]))
				if max(results) == 0:
					args_max = random.randint(0,len(letters))
				else: 
					args_max = results.index(max(results))
				if args_max != letters[item['result_class'].upper()]:
					#print args_max
					#print letters[item['result_class'].upper()]
					#print  "Hello" + str(len(self.weight_vector[args_max*128:args_max*128 + 128]))

					if len(self.weight_vector[args_max*128:args_max*128 + 128]) > 0:
						self.weight_vector[args_max*128:args_max*128 + 128] -= np.array(item['descriptor'])
						val = letters[item['result_class'].upper()]
						self.weight_vector[val*128:val*128 + 128] += np.array(item['descriptor'])
				else:
					count += 1

	def test_multi_class_classifier(self):
		letters = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}
		if len(self.sample_data) == 0 or len(self.weight_vector) == 0:
			return False
		num_mistakes = 0
		for item in self.sample_data:
				results = []
				for j in range(len(letters)):					
					results.append(np.dot(item['descriptor'], self.weight_vector[j*128:j*128 + 128]))
				if max(results) == 0:
					args_max = random.randint(0,len(letters))
				else: 
					args_max = results.index(max(results))
				if args_max == letters[item['result_class'].upper()]:
					#print "Result: " + str(item['result_class'].upper()) + " == " + str(letters.keys()[letters.values().index(args_max)])
					#raw_input("True")
					pass
				else:
					num_mistakes += 1

		return num_mistakes	
			

if __name__ == "__main__":
	import sys

	ocr = OCR_classifier()
	for file in sys.argv[1:-1]:
		print file
		ocr.read_sample_data(file)
	ocr.read_test_data(sys.argv[-1])

	num_mistakes = []
	RANGE = 50
	ocr.setup_binary_weights()
	for i in range(RANGE):
		ocr.train_binary_classifier(i + 1)		
		num_mistakes.append(ocr.test_binary_classifier())
		ocr.setup_binary_weights()

	print num_mistakes
	plt.plot(range(RANGE), num_mistakes)
	#plt.axis([0,50,0,100])
	plt.show()
	
	
	ocr.setup_multi_class_weights()
	#ocr.read_test_data(sys.argv[-1])
	num_mistakes = []
	for i in range(50):	
		ocr.train_multi_class_classifier(i)
		num_mistakes.append(ocr.test_multi_class_classifier())
		ocr.setup_multi_class_weights()

	
	plt.plot(range(50), num_mistakes)
	plt.axis([0,50,0,100])
	plt.show()

