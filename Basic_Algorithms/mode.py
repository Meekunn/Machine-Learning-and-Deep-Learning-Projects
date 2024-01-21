def mode(data):
	max_value = 0
	max_frequency = 0
	current_value = 0
	frequency = 0

	for i in range(len(data)):
		if data[i] == current_value:
			frequency += 1
			print(current_value, frequency)
		else: 
			current_value = data[i]
			frequency = 0
			print(current_value, frequency)

		if frequency > max_frequency:
			max_value = current_value
			max_frequency = frequency
			print(current_value, frequency, max_frequency, max_value)
			# frequency = 0

	return max_value

data = [1,1,1,1,1,2,2,2,2,3,3,4,4]
print("mode: ",mode(data))