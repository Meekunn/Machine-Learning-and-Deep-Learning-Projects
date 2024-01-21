from sklearn import datasets
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class KNN_Classifier :
	def __init__(self, k):
		self.k = k 

	def fit(self, x, y):
		self.x = x 
		self.y = y 

	def __euclidean(self, point_a, point_b):
		summed_squares = 0;
		for i in range(len(point_a)):
			summed_squares += (point_a[i] - point_b[i]) ** 2
		return math.sqrt(summed_squares)

	def predict_for_one(self, node):
		distance_label = []
		for i, d in enumerate(self.x):
			dist = self.__euclidean(d, node)
			l = self.y[i]
			distance_label.append((dist, l))

		for i in range(len(distance_label) - 1):
			for j in range(len(distance_label) - i - 1):
				d1, l1 = distance_label[j]
				d2, l2 = distance_label[j + 1]
				if(d1 > d2):
					distance_label[j], distance_label[j + 1] = distance_label[j + 1], distance_label[j]
		
		k_nearest_neighbors_labels = []
		for i in range(self.k):
			_, l = distance_label[i]
			k_nearest_neighbors_labels.append(l)

		prediction = self.__mode(k_nearest_neighbors_labels)
		return prediction[0]

	def predict(self, two_d_arr):
		predictions = []
		for sample in two_d_arr:
			pred = self.predict_for_one(sample)
			predictions.append(pred)
		
		return np.array(predictions)

	def __mode(self, data): 
		mode = []
		mode_dict = {}
		count = 0

		for elem in data:
			if elem in mode_dict:
				mode_dict[elem] += 1
			else:
				mode_dict[elem] = 0
		sorted_dict = sorted(mode_dict.items(), key=lambda x:x[1], reverse=True) 
		for tup in sorted_dict:
			if(tup[1] == sorted_dict[0][1]):
				mode.append(tup[0])
			else: 
				continue
		return mode
		
	def accuracy(self, ground_truth, prediction):
		accurate_count = 0
		comparison_list = zip(ground_truth, prediction)
		for truth, predict in comparison_list:
			if predict == truth:
				accurate_count += 1
		
		print(accurate_count, ground_truth.shape[0])
		accuracy =  accurate_count / ground_truth.shape[0]
		accuracy_percent = accuracy * 100

		return accuracy_percent


def main():

	iris = datasets.load_iris()
	# print(iris.data.shape)
	_, ax = plt.subplots()
	scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
	ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
	_ = ax.legend(
			scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
	)
	# plt.show()

	dimension_reduc = PCA(n_components=3)
	new_data = dimension_reduc.fit_transform(iris.data)
	# print(new_data.shape)
	_, ax = plt.subplots()
	scatter = ax.scatter(new_data[:, 0], new_data[:, 1], c=iris.target)
	ax.set(xlabel="Feature 1", ylabel="Feature 2")
	_ = ax.legend(
			scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
	)

	# plt.show()
	indexes = np.arange(new_data.shape[0])
	np.random.shuffle(indexes)

	train_percent = 0.1
	num_train_data = int(train_percent * new_data.shape[0])

	train_index = indexes[:num_train_data]
	test_index = indexes[num_train_data:]

	X_train_data = new_data[train_index,:]
	X_test_data = new_data[test_index,:]
	y_train_label = iris.target[train_index]
	y_test_label = iris.target[test_index]
	# print(X_train_data.shape)
	# print(X_test_data.shape)
	# print(y_train_label)
	# print(y_test_label)

	knn = KNN_Classifier(3)
	knn.fit(X_train_data, y_train_label)
	predictions = knn.predict(X_test_data)
	accuracy = knn.accuracy(y_test_label, predictions)
	print(accuracy)


main()

