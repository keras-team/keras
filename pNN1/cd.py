# cd.py Creates a data set for NN that is all spirals and has the ability to make a data set that is a 2d matrix paired with a 1d matrix of 'classes'
# 'classes' as example are like different sensor readinds. Say you got 2 heat sensors giving a nuemerical output every couple seconds, thats one class of data. Say you got 2 other sensors for pH, one in the input flow to a tank and output flow in a tank for example, another class
import numpy as np

np.random.seed(0)


def create_data(points, classes):

    x = np.zeros(
        (points * classes, 2)
    )  # Creates a matrix of zeros that is points*classes number of rows and 2 columns
    # print(x)

    y = np.zeros(points * classes, dtype="uint8")
    # print(y)

    # print(range(classes))

    for class_number in range(
        classes
    ):  # for loop that runs for as many iterations as classes input where class_number starts at 0
        # print(class_number) # prints class number in terminal each loop

        ix = range(
            points * class_number, points * (class_number + 1)
        )  # creates an array of numbers starting at 'points*class_number', where class number is 0,1,2 (when a 3 is input as the number of classes), and 'points*(class_number+1)' is the last number.
        # So for example output on first iteration = 100*0, 100 (list of numbers 0-99)     Next iteration = 100*1, 100*(1+1)  (list of numbers 100-199)    ect...
        # print(ix)

        r = np.linspace(
            0.0, 1, points
        )  # radius. Range is established in first 2 inputs. Points determines the number of values in the 1d matrix that are evenly distributed from first and second input
        # print(r)

        # this is a special guy syntax-wise. Just keek him all together. 'points' is a 1d array that specifies size of the random number distribution, 2 is the standard deviation of that distribution, 0 is the mean
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + (
            np.random.randn(points) * 2 + 0
        )  # Multiples class number by 4 and class number +1 by 4 to determine range of list. 1d array with points as number of rows. Adds normally distributed random number with mean 0 and standard deviation 2 to all values
        # print(t)

        x[ix] = np.c_[
            r * np.sin(t * 2.5), r * np.cos(t * 2.5)
        ]  # Makes a 2d matrix of scaled by 'r' sin wave values that have frequency 2.5 = 't'.   y(t) = A sin(ωt + φ)
        # does the same with cos in 2nd column
        # print(x[ix])

        y[ix] = class_number  # A 1d array of 100 numbers that are class_number.
        # print(y[ix]) #So in this example, 0, 1, 2.

    return x, y


"""

import matplotlib.pyplot as plt
print("here")
x, y = create_data(100,3)
#print(x)

#print(y)
plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')
plt.show()

#print(x, y)
"""
