""" 801021051 SURAJ M SHETTY """
# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
#
# TODO: Write this.
#
# Takes the yx file as input, where on each line y is the first element
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np
from pyspark import SparkContext


def matmulxx(matx):
    # finding x transpose
    matx_transpose = matx.transpose()
    # matrix x transpose multiplied with matrix x
    result = matx * matx_transpose
    return 'keyA', result


def matmulxy(y, x):
    maty = np.asmatrix(y).astype('float')
    result = x * maty
    return 'keyB', result


def get_listx(line):
    line.pop(0)
    line.insert(0, 1)
    t = ""
    # adding number to matrix t
    for num in line:
        t = t + str(num) + ";"
    t = t[:-1]
    # matrix is now type(float)
    return np.matrix(t).astype('float')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: linreg <datafile>"
        exit(-1)

    sc = SparkContext(appName="LinearRegression")

    # Input yx file has y_i as the first element of each line
    # and the remaining elements constitute x_i
    yxinputFile = sc.textFile(sys.argv[1])
    rdd = yxinputFile.map(lambda line: line.split(','))
    yxfirstline = rdd.first()
    yxlength = len(yxfirstline)
    # print "yxlength: ", yxlength

    # dummy floating point array for beta to illustrate desired output format
    beta = np.zeros(yxlength, dtype=float)

    # retrieving the x list and calculating the first term
    xx = rdd.map(lambda line: get_listx(line))
    xx_result = xx.map(matmulxx)
    first_term = xx_result.reduceByKey(lambda x, y: x + y)
    first_term.collect()

    # calculating the second term
    xy = rdd.map(lambda line: matmulxy(line[0], get_listx(line)))
    second_term = xy.reduceByKey(lambda x, y: x + y)
    second_term.collect()

    # retrieving the first matrix value , second matrix and inverse
    matxx = first_term.values().first()
    matxy = second_term.values().first()
    matxx_inv = np.asmatrix(np.linalg.inv(matxx))

    # multiplying xx inverse * matrix xy
    mat_result = matxx_inv * matxy

    beta = np.squeeze(np.asarray(mat_result))

    # print the linear regression coefficients in desired output format
    print "beta: "
    for coeff in beta:
        print coeff

    sc.stop()
