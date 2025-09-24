class car():
    def __init__(self, name, model, year):
        self.name = name
        self.model = model
        self.year = year

    def get_name(self):
        return self.name

    def get_model(self):
        return self.model

    def get_year(self):
        return self.year
def dd():
    print("heevbs")
    return None
 
  
d  = dd()
print(d)
class human():
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def women(self):
        print("hello women",self.name)
        print("age = ",self.age)
    def man(self):
        print("hello man",self.name)
        print("age = ",self.age)
    def display(self):
        print("name = ",self.name)
        print("age = ",self.age)

human2 = human("dsfgnm" , 77 )
d = human2.man()

print(d)
print(human2.name)

print(human2.age)
def greet(name):
       return f"Hello, {name}!"

def add(a, b):
       return a + b







# modules 

import sample2
from sample2 import point
d = sample2
c2 = point()
c2.draw()
print(c2.draw())