


import numpy as np

solver = "BO"



if solver == "BO":
	print("It's BO")



if solver == "BO":
	print("It's BO again.")
elif sovler == "PRS":
	print("It's pure Random Search")
else:
	raise ValueError("It's should be PRS or BO")


def fun(x):
	if x is None:
		print("It's none.")
	else:
		print(x)

fun(None)