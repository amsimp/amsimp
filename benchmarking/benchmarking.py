# -----------------------------------------------------------------------------------------#

# Importing Dependencies
import amsimp
import time
import csv
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------#

filename = raw_input("The name of the CSV file: ")


def csv_file():
    file = os.path.isfile(filename)
    csvfile = open(filename, "a")

    fieldnames = ["detail_level", "time"]
    writer = csv.DictWriter(
        csvfile, delimiter=",", lineterminator="\n", fieldnames=fieldnames
    )

    if not file:
        writer.writeheader()

    return writer


def write_data(writer, data):
    writer.writerow(data)


# -----------------------------------------------------------------------------------------#

samples = int(input("The number of samples for each level of detail: "))

if not isinstance(samples, int):
    raise Exception(
        "samples must be an integer. The value of detail_level was: {}".format(
            self.detail_level
        )
    )

if samples < 1:
    raise Exception(
        "samples must be a positive integer. The value of detail_level was: {}".format(
            self.detail_level
        )
    )

# -----------------------------------------------------------------------------------------#


def benchmarking(samples):
    writer = csv_file()
    for i in range(samples):
        for num in range(5):
            start = time.time()
            detail = amsimp.Dynamics(num + 1)
            detail.simulate(True)
            plt.close("all")
            finish = time.time()
            t = finish - start
            write_data(writer, {"detail_level": num + 1, "time": t})
        print(
            "Progress: "
            + str(i + 1)
            + " samples have been run out of a total of "
            + str(samples)
        )
    print("Benchmarking Complete.")


# -----------------------------------------------------------------------------------------#

benchmarking(samples)