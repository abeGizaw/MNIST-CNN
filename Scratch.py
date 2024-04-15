import numpy as np

test1 = np.array([[1,1,1],
                  [-2,1,-1]])


print(test1.shape)
print(np.maximum(0, test1))

din = []
din[test1 <= 0] = 0

test2 = np.array([[[1,1,1],
                  [-2,1,-1]],[[1,1,1],
                  [-2,1,-1]],[[1,1,1],
                  [-2,1,-1]],[[1,1,1],
                  [-2,1,-1]]])

print(test2.shape)
print(np.maximum(0, test2))