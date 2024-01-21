##  The KNN Algorithm
# 1.  Load the data
# 2.  Initialize K to your chosen number of neighbors
# 3. For each example in the data
# 		3.1 Calculate the distance between the query example and the current example from the data.
# 		3.2 Add the distance and the index of the example to an ordered collection
# 4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
# 5. Pick the first K entries from the sorted collection
# 6. Get the labels of the selected K entries
# 7. If regression, return the mean of the K labels
# 8. If classification, return the mode of the K labels

# X is a plot (x,y)
# Y is the label Y
import math
import numpy as np
import matplotlib.pyplot as plt

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

	def predict(self, node):
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

	def closest_points(self, node):
		distances = []

		for i in range(len(self.x)):
			distances.append(self.__euclidean(self.x[i], node))

		sorted_distances = sorted(distances)

		k_nearest_neighbors_indices = []
		for i in range(self.k):
			for j in range(len(self.x)):
				if (sorted_distances[i] == distances[j]):
					k_nearest_neighbors_indices.append(j)
					break
		
		k_nearest_neighbors_labels = []
		for i in range(self.k):
			k_nearest_neighbors_labels.append(self.y[k_nearest_neighbors_indices[i]])

		prediction = self.__mode(k_nearest_neighbors_labels)
		return prediction[0]

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
		

def main():

	#data = [2,2,3,1,3,4,1,8,4,9]
	x_cords = np.random.randint(0, 15, 20)
	y_cords = np.random.randint(0, 15, 20)

	X = list(zip(x_cords,y_cords))
	Y = np.random.randint(0, 2, 20)

	X_0 = [X[i] for i in range(len(X)) if Y[i] == 0]
	X_1 = [X[i] for i in range(len(X)) if Y[i] == 1]	
	
	knn = KNN_Classifier(3)
	knn.fit(X, Y)
	print(knn.predict((6, 4.2)))
	print(knn.closest_points(((6, 4.2))))
	print(knn.closest_points(((6, 4.2))))

	plt.scatter(*zip(*X_0), color='red', label='Y=0')
	plt.scatter(*zip(*X_1), color='blue', label='Y=1')

	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')

	plt.legend()
	plt.show()


main()