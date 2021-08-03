import numpy as np
import math
import os
import time
import random
import gc
from sko.GA import GA
from hyperopt import fmin, tpe, hp


grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = os.path.join(grandfather_path, "Calculations", "Auto_deep_learning")
if not os.path.exists(path):
    os.mkdir(path)


def shaffer_F6(x, y) -> object:
    f1 = math.sin(math.sqrt(pow(x, 2) + pow(y, 2)))
    f2 = pow(f1, 2) - 0.5
    f3 = pow(1 + 0.001 * (pow(x, 2) + pow(y, 2)), 2)
    f4 = 0.5 - f2 / f3
    return f4


class QGA(object):
    def __init__(self, population_size=200, chromosome_num=2, chromosome_length=20, max_value=100, min_value=1,
                 iter_nums=100, theta=0.01*np.pi):
        self.population_size = population_size
        self.chromosome_num = chromosome_num
        self.chromosome_length = chromosome_length
        self.max_value = max_value
        self.min_value = min_value
        self.iter_nums = iter_nums
        self.theta = theta

    def train_step(self):
        results = []
        best_fitness = 0
        best_parameter = [0, 0]
        population_Angle = self.species_origin_angle()

        layers_files = os.path.join(path, "x_y_best_fitness.txt")
        if os.path.exists(layers_files):
            os.remove(layers_files)

        fitness_files = os.path.join(path, "xy_best_fitness.txt")
        if os.path.exists(fitness_files):
            os.remove(fitness_files)

        for iter_num in np.arange(self.iter_nums):
            start_time = time.time()
            population_Q = self.population_Q(population_Angle)
            population_Binary = self.translation(population_Q)
            parameters, fitness_value, current_parameter_Binary, current_fitness, current_parameter \
                = self.fitness(population_Binary)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_parameter = current_parameter
            print('iteration is :', iter_num + 1, ';Best parameters:', best_parameter, ';Best fitness', best_fitness)
            results.append(best_fitness)

            population_Angle_crossover = self.crossover(population_Angle)
            population_Angle = self.mutation(population_Angle_crossover, population_Angle, current_parameter_Binary,
                                             current_fitness)

            end_time = time.time()
            train_time = end_time - start_time

            filename = os.path.join(path, "xy_best_fitness.txt")
            with open(filename, "a+") as f:
                f.write(str(iter_num) + "," + str(best_parameter[0]) + "," + str(best_parameter[1])
                        + "," + str(best_fitness) + "," + str(train_time) + "\n")
        gc.collect()

    def species_origin_angle(self):
        population_Angle = np.empty((self.chromosome_num, self.population_size, self.chromosome_length))
        for i_index in np.arange(self.chromosome_num):
            for j_index in np.arange(self.population_size):
                for k_index in np.arange(self.chromosome_length):
                    population_Angle[i_index][j_index][k_index] = np.pi * 2 * random.random()
        return population_Angle

    def population_Q(self, population_Angle):
        population_Q = np.empty((self.chromosome_num, self.population_size, 2, self.chromosome_length))
        for i_index in np.arange(self.chromosome_num):
            for j_index in np.arange(self.population_size):
                for m_index in np.arange(self.chromosome_length):
                    theta = population_Angle[i_index][j_index][m_index]
                    population_Q[i_index][j_index][0][m_index] = np.sin(theta)
                    population_Q[i_index][j_index][1][m_index] = np.cos(theta)
        return population_Q

    def translation(self, population_Q):
        population_Binary = np.empty((self.chromosome_num, self.population_size, self.chromosome_length))
        for i_index in np.arange(self.chromosome_num):
            for j_index in np.arange(self.population_size):
                for k_index in np.arange(self.chromosome_length):
                    if np.square(population_Q[i_index][j_index][0][k_index]) > random.random():
                        population_Binary[i_index][j_index][k_index] = 1
                    else:
                        population_Binary[i_index][j_index][k_index] = 0
        return population_Binary

    def fitness(self, population_Binary):
        parameters = np.empty((self.chromosome_num, self.population_size))
        for i_index in np.arange(self.chromosome_num):
            for j_index in np.arange(self.population_size):
                total = 0.0
                for k_index in np.arange(self.chromosome_length):
                    total += population_Binary[i_index][j_index][k_index] * math.pow(2, k_index)
                value = (total * (self.max_value - self.min_value)) / math.pow(2,
                                                                               self.chromosome_length) + self.min_value
                parameters[i_index][j_index] = value

        fitness_value = []
        for i_index in np.arange(self.population_size):
            x = parameters[0][i_index]
            y = parameters[1][i_index]
            score = shaffer_F6(x, y)
            fitness_value.append(score)

            filename = os.path.join(path, "x_y_best_fitness.txt")
            with open(filename, "a+") as f:
                f.write(str(x) + "," + str(y) + "," + str(score) + "\n")

        best_fitness = 0
        best_parameter = np.empty(self.chromosome_num)
        best_parameter_Binary = np.empty((self.chromosome_num, self.chromosome_length))

        for i_index in np.arange(self.population_size):
            if best_fitness < fitness_value[i_index]:
                best_fitness = fitness_value[i_index]
                for j_index in np.arange(self.chromosome_num):
                    best_parameter[j_index] = parameters[j_index][i_index]
                    for k_index in np.arange(self.chromosome_length):
                        best_parameter_Binary[j_index][k_index] = population_Binary[j_index][i_index][k_index]

        return parameters, fitness_value, best_parameter_Binary, best_fitness, best_parameter

    def crossover(self, population_Angle):
        population_Angle_crossover = np.empty((self.chromosome_num, self.population_size, self.chromosome_length))
        for i_index in np.arange(self.chromosome_num):
            for j_index in np.arange(self.population_size):
                for k_index in np.arange(self.chromosome_length):
                    ni = (j_index - k_index) % self.population_size
                    population_Angle_crossover[i_index][j_index][k_index] = population_Angle[i_index][ni][k_index]
        return population_Angle_crossover

    def mutation(self, population_Angle_crossover, population_Angle, best_parameter_Binary, best_fitness):
        population_Q_crossover = self.population_Q(population_Angle_crossover)
        population_Binary_crossover = self.translation(population_Q_crossover)
        parameters, fitness_crossover, best_parameter_Binary_crossover, best_fitness_crossover, best_parameter = \
            self.fitness(population_Binary_crossover)

        Rotation_Angle = np.empty((self.chromosome_num, self.population_size, self.chromosome_length))

        for i_index in range(self.chromosome_num):
            for j_index in range(self.population_size):
                if fitness_crossover[j_index] <= best_fitness:
                    for k_index in range(self.chromosome_length):
                        s1 = 0
                        a1 = population_Q_crossover[i_index][j_index][0][k_index]
                        b1 = population_Q_crossover[i_index][j_index][1][k_index]
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a1 * b1 == 0:
                            s1 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a1 * b1 == 0:
                            s1 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a1 * b1 == 0:
                            s1 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a1 * b1 == 0:
                            s1 = 1
                        Rotation_Angle[i_index][j_index][k_index] = self.theta * s1
                else:
                    for k_index in np.arange(self.chromosome_length):
                        s2 = 0
                        a2 = population_Q_crossover[i_index][j_index][0][k_index]
                        b2 = population_Q_crossover[i_index][j_index][1][k_index]
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a2 * b2 > 0:
                            s2 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a2 * b2 < 0:
                            s2 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a2 * b2 > 0:
                            s2 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a2 * b2 < 0:
                            s2 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 0 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a2 * b2 > 0:
                            s2 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a2 * b2 < 0:
                            s2 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a2 * b2 > 0:
                            s2 = 1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a2 * b2 < 0:
                            s2 = -1
                        if population_Binary_crossover[i_index][j_index][k_index] == 1 and \
                                best_parameter_Binary[i_index][k_index] == 1 and a2 * b2 == 0:
                            s2 = 1
                        Rotation_Angle[i_index][j_index][k_index] = self.theta * s2

        for i_index in range(self.chromosome_num):
            for j_index in range(self.population_size):
                for k_index in range(self.chromosome_length):
                    population_Angle[i_index][j_index][k_index] = population_Angle[i_index][j_index][k_index] + \
                                                                  Rotation_Angle[i_index][j_index][k_index]

        return population_Angle


file_xy = os.path.join(path, "GA_xy_fitness.txt")


def GA_shaffer_F6(x, y) -> object:
    f1 = math.sin(math.sqrt(pow(x, 2) + pow(y, 2)))
    f2 = pow(f1, 2) - 0.5
    f3 = pow(1 + 0.001 * (pow(x, 2) + pow(y, 2)), 2)
    f4 = 0.5 - f2 / f3

    with open(file_xy, "a+") as f:
        f.write(str(x) + "," + str(y) + "," + str(f4) + "\n")

    return -f4


def GA_func():
    if os.path.exists(file_xy):
        os.remove(file_xy)
    ga = GA(func=GA_shaffer_F6, n_dim=2, size_pop=50, max_iter=5001, lb=[-10, -10], ub=[1, 1], precision=1e-7)
    best_x, best_y = ga.run()

    file = os.path.join(path, "GA_x_y_fitness.txt")
    with open(file, "w+") as f:
        f.write("best_x:" + str(best_x[0]) + "," + str(best_x[1]) + "\t" + "best_y: " + str(best_y[0]) + "\n")
        for i_index in range(5001):
            f.write(str(ga.generation_best_X[i_index][0]) + "," + str(ga.generation_best_X[i_index][1]) + ","
                    + str(ga.generation_best_Y[i_index]) + "\n")


Bayes_file_xy = os.path.join(path, "Bayes_xy_fitness.txt")


def Bayes_shaffer_F6(space) -> object:
    f1 = math.sin(math.sqrt(pow(space['x'], 2) + pow(space['y'], 2)))
    f2 = pow(f1, 2) - 0.5
    f3 = pow(1 + 0.001 * (pow(space['x'], 2) + pow(space['y'], 2)), 2)
    f4 = 0.5 - f2 / f3

    with open(Bayes_file_xy, "a+") as f:
        f.write(str(space['x']) + "," + str(space['y']) + "," + str(f4) + "\n")

    return -f4


def Bayes_func():
    if os.path.exists(Bayes_file_xy):
        os.remove(Bayes_file_xy)
    best = fmin(fn=Bayes_shaffer_F6, space={'x': hp.uniform('x', -10, 10), 'y': hp.uniform('y', -10, 10)},
                algo=tpe.suggest, max_evals=5001)

    file = os.path.join(path, "Bayes_x_y_fitness.txt")
    with open(file, "w+") as f:
        f.write("best_x:" + str(best['x']) + "," + str(best['y']) + "\n")


Random_file_xy = os.path.join(path, "Random_xy_fitness.txt")


def Random_shaffer_F6(x, y) -> object:
    f1 = math.sin(math.sqrt(pow(x, 2) + pow(y, 2)))
    f2 = pow(f1, 2) - 0.5
    f3 = pow(1 + 0.001 * (pow(x, 2) + pow(y, 2)), 2)
    f4 = 0.5 - f2 / f3

    with open(Random_file_xy, "a+") as f:
        f.write(str(x) + "," + str(y) + "," + str(f4) + "\n")

    return f4


def Random_func():
    iteration = 5001
    x_list = np.arange(-10, 10, 0.001)
    y_list = np.arange(-10, 10, 0.001)
    best_x = 0.0
    best_y = 0.0
    best_fitness = 0.0

    for _ in np.arange(iteration):
        x = random.choice(x_list)
        y = random.choice(y_list)
        fitness = Random_shaffer_F6(x, y)
        if fitness > best_fitness:
            best_x = x
            best_y = y
            best_fitness = fitness

    file = os.path.join(path, "Random_x_y_fitness.txt")
    with open(file, "w+") as f:
        f.write("best_x: " + str(best_x) + "," + "best_y: " + str(best_y) + "," + "fitness: " + str(best_fitness) + "\n")


def GA_shaffer_F6_time(x, y) -> object:
    f1 = math.sin(math.sqrt(pow(x, 2) + pow(y, 2)))
    f2 = pow(f1, 2) - 0.5
    f3 = pow(1 + 0.001 * (pow(x, 2) + pow(y, 2)), 2)
    f4 = 0.5 - f2 / f3

    return -f4


def GA_func_time():
    file = os.path.join(path, "GA_time.txt")
    if os.path.exists(file):
        os.remove(file)

    ga = GA(func=GA_shaffer_F6_time, n_dim=2, size_pop=50, max_iter=5001, lb=[-10, -10], ub=[1, 1], precision=1e-7)
    for _ in np.arange(5001):
        start_time = time.time()
        ga.run(1)
        end_time = time.time()
        ga_time = end_time - start_time

        with open(file, "a") as f:
            f.write(str(ga_time) + "\n")
