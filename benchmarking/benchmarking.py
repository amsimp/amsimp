#-----------------------------------------------------------------------------------------#

#Importing Dependencies
import amsimp
import time
import csv
import os
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------#

filename = 'benchmarking_data.csv'

def csv_file():
	file = os.path.isfile(filename)
	csvfile = open(filename, 'a')

	fieldnames = ['detail_level', 'time']
	writer = csv.DictWriter(csvfile, delimiter = ',', lineterminator = '\n', fieldnames = fieldnames)

	if not file:
		writer.writeheader()
		
	return writer

def write_data(writer, data):
	writer.writerow(data)

#-----------------------------------------------------------------------------------------#

samples = int(input('The number of samples for each level of detail: '))

#-----------------------------------------------------------------------------------------# 

def benchmarking(samples):
	writer = csv_file()
	for i in range(samples):
		for num in range(4):
			start = time.time()
			detail = amsimp.Dynamics(num + 2)
			detail.simulate(True)
			plt.close('all')
			finish = time.time()
			t = finish - start
			write_data(writer, {'detail_level': num + 2, 'time': t})

#-----------------------------------------------------------------------------------------#

benchmarking(samples)