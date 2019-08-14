"""
My Library for linear Algebra/ Matrix math for machine learning purposes
Create Matrices either with data or with number of columns, rows and matrix initialisation type
Example 1:
    data1 = [[3, 5, 1, 6],
            [2, 3, 6, 7],
            [5, 2, 5, 3],
            [2, 8, 7, 7]]
    A = Matrix(data=data1)
This returns a Matrix object with the dimensions and values as given in the data:

Example 2:
    A = Matrix(3, 6, mat_type="random")
This returns a Matrix object with 3 rows and 6 columns and random values between -1 an 1.

Matrix types are:
    "random": random values between -1 and 1
    "zeros": all zero matrix
    "gauss_random": random normal distribution values with mu=0 and sigma=columns ** (-1/2)

Things that should be added in the future for neuro-evolution maybe in a subclass called neuro-evol:
    def crossover
    def mutation
    def inverse
"""
import random
import math


class Matrix:
    # initialize the Matrix object either with data or random with given rows and columns
    def __init__(self, rows=0, cols=0, mat_type="random", data=None):
        if data:
            # if isinstance(data, list):
            #    self.rows = len(data)
            #    self.cols = 1
            #    self.zeros()
            #    for i in range(len(data)):
            #        self.matrix_data[i][0] += data[i]
            self.matrix_data = data
            self.rows = len(data)
            self.cols = len(data[0])
        else:
            self.rows = rows
            self.cols = cols
            self.mat_type(mat_type)

    # show the Matrix object
    @staticmethod
    def print_mat(a):
        if a:
            if type(a) is Matrix:
                for i in range(a.rows):
                    print(a.matrix_data[i])
            else:
                print(">>> No data in Matrix!")
        else:
            print(">>> Not object specified!")

    # multiply two Matrix objects together
    @staticmethod
    def mat_mul(a, b):
        c = Matrix(a.rows, b.cols, mat_type="zeros")
        if a.cols != b.rows:
            print(">>> The two Matrices are not compatible with this operation!")
            return None
        else:
            for i in range(a.rows):
                for j in range(b.cols):
                    for k in range(b.rows):
                        c.matrix_data[i][j] += a.matrix_data[i][k] * b.matrix_data[k][j]
            return c

    # element wise addition of two matrices
    @staticmethod
    def mat_add(a, b):
        c = Matrix(a.rows, b.cols, mat_type="zeros")
        # if A.cols != B.rows:
        #     print(">>> The two Matrices are not compatible with this operation!")
        #     return None
        # else:
        for i in range(a.rows):
            for j in range(b.cols):
                c.matrix_data[i][j] += a.matrix_data[i][j] + b.matrix_data[i][j]
        return c

    # make a matrix object with specific type
    def mat_type(self, mat_type):
        if mat_type == "random":
            self.randomize()
        if mat_type == "zeros":
            self.zeros()
        if mat_type == "gauss_random":
            self.gauss_random()

    # apply a function to every element of the matrix (for activation function)
    @staticmethod
    def apply_func(a, func):
        b = Matrix(a.rows, a.cols, mat_type="zeros")
        for i in range(a.rows):
            for j in range(a.cols):
                b.matrix_data[i][j] += func(a.matrix_data[i][j])
        return b

    # create a random matrix with gaussian distribution
    def gauss_random(self):
        self.matrix_data = [[random.gauss(0, self.cols ** (-1 / 2)) for x in range(self.cols)] for y in range(self.rows)]

    # create a random matrix with random distribution
    def randomize(self):
        self.matrix_data = [[random.uniform(-1, 1) for x in range(self.cols)] for y in range(self.rows)]

    # create matrix with all values equal to zero
    def zeros(self):
        self.matrix_data = [[0 for x in range(self.cols)] for y in range(self.rows)]

    # copy a matrix
    @staticmethod
    def copy(a):
        b = Matrix(a.rows, a.cols, mat_type="zeros")
        for i in range(a.rows):
            for j in range(a.cols):
                b.matrix_data[i][j] += a.matrix_data[i][j]
        return b

    # create a vector (one dimensional matrix)
    @staticmethod
    def create_vector(data):
        new_vector = Matrix(len(data), 1, mat_type="zeros")
        for i in range(new_vector.rows):
            new_vector.matrix_data[i][0] = data[i]
        return new_vector

    # element wise addition of two vectors
    @staticmethod
    def vector_add(a, other):
        new_vector = Matrix(a.rows, 1, mat_type="zeros")
        if a.rows is not other.rows:
            print(">>> Cannot add. Objects are not Vectors or have different dimensions!")
            return None
        for i in range(a.rows):
            new_vector.matrix_data[i][0] += other.matrix_data[i][0] + a.matrix_data[i][0]
        return new_vector

    # element wise subtraction of two vectors
    @staticmethod
    def vector_sub(a, other):
        new_vector = Matrix(a.rows, 1, mat_type="zeros")
        if a.rows is not other.rows:
            print(">>> Cannot sub. Objects are not Vectors or have different dimensions!")
            return None
        for i in range(a.rows):
            new_vector.matrix_data[i][0] += other.matrix_data[i][0] - a.matrix_data[i][0]
        return new_vector

    # element wise multiplication of two vectors
    @staticmethod
    def vector_mult(a, other):
        new_vector = Matrix(a.rows, 1, mat_type="zeros")
        if a.rows is not other.rows:
            print(">>> Cannot multiply. Objects are not Vectors or have different dimensions!")
            return None
        for i in range(a.rows):
            new_vector.matrix_data[i][0] += other.matrix_data[i][0] * a.matrix_data[i][0]
        return new_vector

    # add a scalar to a matrix (element wise)
    @staticmethod
    def add(a, x):
        b = Matrix(a.rows, a.cols, mat_type="zeros")
        for i in range(a.rows):
            for j in range(a.cols):
                b.matrix_data[i][j] += a.matrix_data[i][j] + x
        return b

    @staticmethod
    def mult(a, x):
        b = Matrix(a.rows, a.cols, mat_type="zeros")
        for i in range(a.rows):
            for j in range(a.cols):
                b.matrix_data[i][j] += a.matrix_data[i][j] * x
        return b

    @staticmethod
    def sub(a, x):
        b = Matrix(a.rows, a.cols, mat_type="zeros")
        for i in range(a.rows):
            for j in range(a.cols):
                b.matrix_data[i][j] += a.matrix_data[i][j] - x
        return b

    def soft_max(self):
        if self.cols is not 1:
            print(">>> Cannot soft-max Vector object!")
            return None
        else:
            summation = 0
            for item in self.matrix_data:
                summation += math.e ** item[0]
            for i in range(self.rows):
                self.matrix_data[i][0] = (math.e ** self.matrix_data[i][0]) / summation
            return self

    @staticmethod
    def transpose(a):
        new_rows = a.cols
        new_cols = a.rows
        new_matrix = Matrix(new_rows, new_cols, mat_type="zeros")
        for i in range(a.rows):
            for j in range(a.cols):
                new_matrix.matrix_data[j][i] = a.matrix_data[i][j]
        return new_matrix

    def arg_max(self):
        data_list = []
        for i in range(self.rows):
            for j in range(self.cols):
                data_list.append(self.matrix_data[i][j])
        return data_list.index(max(data_list))
