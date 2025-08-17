import math
weight = float(input("enter a wight : "))
unit = input("kilogram or pouds (k / p) : ")

if unit == "k":
    weight = weight * 2.205
    unit = "lbs"

elif unit == "p" :
    weight = weight / 2.205
    unit = "kgs"
      
else:
    print(f"{unit} is not valid")

    print (f"your weight  is :{round(weight)} ")