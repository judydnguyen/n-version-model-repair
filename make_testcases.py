import csv
import sys
import random

LINE_LEN = 7
MIN_IDX = 10

# pass in the controller number
bad = sys.argv[1]

# max_idx = int(sys.argv[2])

with open("missed_" + bad + ".txt") as missed:
    miss = missed.readlines()
    miss = [x[:-1] for x in miss]
    print("controller missed: " + str(miss))

with open('results.csv', 'r') as csvfile, open('passing_cases.csv', 'w') as passing, open('failing_cases.csv', 'w') as failing:
    csv_reader = csv.reader(csvfile)
    pass_writer = csv.writer(passing)
    fail_writer = csv.writer(failing)

    pass_writer.writerow(["idx", "state0", "state1", "state2", "state3", "user_input", "controller_vote", "actuation"])
    fail_writer.writerow(["idx", "state0", "state1", "state2", "state3", "user_input", "controller_vote", "actuation"])


    for row in csv_reader:
        if len(row) != LINE_LEN:
            continue
        idx = row[0]
        if idx in miss:
            vote = "0"
            if row[6] == "0":
                vote = "1"
            fail_writer.writerow(row[:6]+ [vote, row[6]])
            #print("failing")
        else:
            #print("passing")
            vote = "0"
            if row[6] == "1":
                vote = "1"
            pass_writer.writerow(row[:6]+ [vote, row[6]])