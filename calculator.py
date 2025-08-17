
operator = input("enter any operator( + - * /) :")
num1 = float(input("enter a 1st number :"))
num2 = float(input("enter a 2nd number : "))

if operator == "+":
    result =num1 + num2
    print(round(result, 3))


elif operator == "-":
    result = num1 - num2
    print(round(result, 3))
elif operator == "*":
    result = num1* num2
    print(round(result, 3))
elif operator == "/":
    result = num1 / num2
    print(round(result, 3))

else:
    print(f"{operator}not vaild")
