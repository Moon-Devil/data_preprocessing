from shaffer_F6_QGA import *

# qga = QGA(50, 2, 20, 10, -10, 5001, 0.01*np.pi)
#
# start_time = time.time()
# qga.train_step()
# end_time = time.time()
# qga_time = end_time - start_time
#
# start_time = time.time()
# GA_func()
# end_time = time.time()
# GA_time = end_time - start_time
#
# start_time = time.time()
# Bayes_func()
# end_time = time.time()
# Bayes_time = end_time - start_time
#
# start_time = time.time()
# Random_func()
# end_time = time.time()
# Random_time = end_time - start_time
#
# file = os.path.join(path, "time.txt")
# with open(file, "w+") as f:
#     f.write("qga: " + str(qga_time) + "," + "GA: " + str(GA_time) + "," + "Bayes: " + str(Bayes_time) + ","
#             + "Random: " + str(Random_time) + "\n")

GA_func_time()

print("done...")
