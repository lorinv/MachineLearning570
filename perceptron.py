#------------------------------------------------------------------------------
# perceptron.py
#------------------------------------------------------------------------------
# A generalized Perceptron machine learning algorithm. The file includes the 
# standard Perceptron algorithm, the Passive-Agressive learning rate, and the
# averaged perceptron version.
#------------------------------------------------------------------------------
# Author: Lorin Vandegrift (lt_vandegrift at live dot com)
# Copyright (c) 2016, Lorin Vandegrift
#------------------------------------------------------------------------------

import random
import numpy as np
import matplotlib.pyplot as plt
import json

class Perceptron:

	def __init__(self, config_path, update_type = "PA"):
		'''
		Constructor for the Perceptron class
		config : string - file to describe classes
		learning_rate_type : string - pass in "PA" to use the
			passive agressive learning rate, pass anything else to 
			use the standard Perceptron learning rate of 1.
		'''
		self.sample_data = []
		self.weight_vector = []
		self.test_data = []		
		self.bias = 0		

		if len(config_path) > 0:
			with open(config_path) as config:
				data = json.load(config)				
				self.classes = data['data']		

		if update_type == "PA":
			self.UPDATE_TYPE = 1
		else:
			self.UPDATE_TYPE = 0


	##TODO: COMBINE THESE
	def set_sample_data(self, sample_data):
		'''
		Pass in the sample data for the classifier to be trained on. 
		Sample data must be in the format: [{'descriptor':[], 'class':value}, ...]
		Return True - successfully passes in the sample data with at least one sample
		Return False - did not successfully pass in the sample_data
		'''
		if type(sample_data) == type([]):
			if len(sample_data) > 0 and type(sample_data[0]) == type({}):
				if 'descriptor' in sample_data[0].keys() and 'class' in sample_data[0].keys():
					self.sample_data = sample_data
					self.weight_vector = [0] * len(self.sample_data[0]['descriptor']) * len(self.classes)										
					return True

		return False

	##TODO: COMBINE THESE
	def set_test_data(self, sample_data):
		'''
		Pass in the sample data for the classifier to be trained on. 
		Sample data must be in the format: [{'descriptor':[], 'class':value}, ...]
		Return True - successfully passes in the sample data with at least one sample
		Return False - did not successfully pass in the sample_data
		'''
		if type(sample_data) == type([]):
			if len(sample_data) > 0 and type(sample_data[0]) == type({}):
				if 'descriptor' in sample_data[0].keys() and 'class' in sample_data[0].keys():
					self.test_data = sample_data					
					return True

		return False

	def generate_learning_rate(self, sample, y_predicted, y_training, num_features):
		'''
		Generate the requested learning rate, either Passive Agressive or Standard Perceptron.
		sample : list - specific feature vector we are currently working with
		y_predicted : string - 
		'''
		if self.UPDATE_TYPE == 1:
			w_training = self.weight_vector[y_training*num_features:y_training*num_features + num_features]
			x_training = sample
			w_predicted = self.weight_vector[y_predicted*num_features:y_predicted*num_features + num_features]
			x_predicted = sample
			final = np.append(x_training, x_predicted)			
			return (1 - (np.dot(w_training,x_training) - np.dot(w_predicted,x_predicted))) / np.linalg.norm(final)**2 
		else:
			return 1

	def get_class_num(self, value):
		'''
		Find which class a given value corresponds to.
		value : string - value to lookup
		'''
		if type(value) == type(""):
			value = value.upper()
		for c in self.classes:
			if value in c['class']:
				return self.classes.index(c)

		return -1		

	def train_classifier(self, iter_num):
		'''
		Train the data using the standard perceptron algorithm. If the Passive Agressive flag is set, 
		use the Passive Agressive update function.
		iter_num : int - number of times to iterate through the data
		'''
		if len(self.sample_data) == 0 or len(self.weight_vector) == 0:
			return False
		for i in range(iter_num):
			random.shuffle(self.sample_data)
			for item in self.sample_data:
				results = []
				num_features = len(item['descriptor'])
				for j in range(len(self.classes)):										
					results.append(np.dot(item['descriptor'], self.weight_vector[j*num_features:j*num_features + num_features]))
				if max(results) == 0:
					args_max = random.randint(0,len(self.classes))
				else:
					args_max = results.index(max(results))
				if args_max != self.get_class_num(item['class']):
					if len(self.weight_vector[args_max*num_features:args_max*num_features + num_features]) > 0:
						learning_rate = self.generate_learning_rate(item["descriptor"], args_max, self.get_class_num(item['class']), num_features) 
						self.weight_vector[args_max*num_features:args_max*num_features + num_features] -= np.array(item['descriptor']) * learning_rate
						val = self.get_class_num(item['class'])
						self.weight_vector[val*num_features:val*num_features + num_features] += np.array(item['descriptor'])
		print self.weight_vector			
		return True
					
	def test_classifier(self):
		if len(self.sample_data) == 0 or len(self.weight_vector) == 0:
			return False
		mistakes = 0		
		for item in self.sample_data:
			results = []
			num_features = len(item['descriptor'])
			for j in range(len(self.classes)):										
				results.append(np.dot(item['descriptor'], self.weight_vector[j*num_features:j*num_features + num_features]))
			args_max = results.index(max(results))
			if args_max != self.get_class_num(item['class']):
				mistakes += 1

		return mistakes

class Averaged_Perceptron(Perceptron):
	def train_classifier(self, iter_num):
		self.c_weight_vector = self.weight_vector
		self.c_bias = 0
		if len(self.sample_data) == 0 or len(self.weight_vector) == 0:
			return False
		for i in range(iter_num):
			random.shuffle(self.sample_data)
			for item in self.sample_data:
				results = []
				num_features = len(item['descriptor'])
				for j in range(len(self.classes)):										
					results.append(np.dot(item['descriptor'], self.weight_vector[j*num_features:j*num_features + num_features]))
				if max(results) == 0:
					args_max = random.randint(0,len(self.classes))
				else:
					args_max = results.index(max(results))
				if args_max != self.get_class_num(item['class']):
					if len(self.weight_vector[args_max*num_features:args_max*num_features + num_features]) > 0:
						learning_rate = self.generate_learning_rate(item["descriptor"], args_max, self.get_class_num(item['class']), num_features) 
						self.weight_vector[args_max*num_features:args_max*num_features + num_features] -= np.array(item['descriptor']) * learning_rate
						val = self.get_class_num(item['class'])
						self.weight_vector[val*num_features:val*num_features + num_features] += np.array(item['descriptor']) 				
						count = 0
				else:
					count += 1


		return True



#*************************************************************************************88

def read_data(file_path):
	test_data = []
	try:
		with open(file_path, 'r') as workfile:	
			for line in workfile:
				line_parts = line.split('\t')				
				if len(line_parts[0]) > 1:						
					descriptor = [int(i) for i in line_parts[1][2:]]				
					result_class = line_parts[2]					
					test_data.append({'descriptor':descriptor, 'class':result_class})				
			return test_data
	except:
		return False

def compute_accuracy_curve(num_mistakes, total_samples, num_iterations):
	plt.plot(range(num_iterations), (total_samples - np.array(num_mistakes))  / float(total_samples))
	plt.show()

def compute_mistake_curve(num_mistakes, num_iterations):
	plt.plot(range(num_iterations), num_mistakes)
	plt.show()	

if __name__ == "__main__":
	import sys
	p = Perceptron(sys.argv[1])
	num_mistakes = []
	num_iterations = 50
	sample_data = []
	for i in range(num_iterations):
		sample_data = read_data(sys.argv[2])	
		if not p.set_sample_data(sample_data):
			print 'Invalid format'
		p.train_classifier(i + 1)
		test_data = read_data(sys.argv[3])	
		p.set_test_data(test_data)
		num_mistakes.append(p.test_classifier())
	compute_mistake_curve(num_mistakes, num_iterations)
	compute_accuracy_curve(num_mistakes, len(sample_data), num_iterations)




