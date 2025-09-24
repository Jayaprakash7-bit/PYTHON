def max_numbe(numbers):
    numbers=[23,45,67,67,7654]
    max = numbers[0]
    for number in numbers:
      if number > max:
         max = number
         return max
i = max_numbe(numbers=[2,4,5,67,567])
print(i)
def min_number(numbers):
    
    min = numbers[0]
    for number in numbers:
        if number < min:
            min = number

    print(min)
def max_number(numbers):
    max = numbers[0]
    for number in numbers:
        if number > max:
            max = number
    return max
