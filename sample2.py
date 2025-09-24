class point():
    def __init__(self , x, y):
        self.x = x
        self.y = y
    def paint(self):
        print("paint")
    def draw(self):
        print("draw")

pro = point(10,30)
pro.y=11
print(pro.y)
print(pro.draw())
print(pro.paint())



class person:
    def __init__(self,name,age,speaklanuage):
       self.name = name
       self.age = age
       self.speaklanuage = speaklanuage
    def display(self):
        print("name : ",self.name )
        print("age = ",self.age)
        print("speaklanuage = ",self.speaklanuage)
    def update(self,name,age,speaklanuage):
        self.name = name 
        self.age = age
        self.speaklanuage = speaklanuage
        print("update successfully")
        print("name : ",self.name )
        print("age = ",self.age)    
        print("speaklanuage = ",self.speaklanuage)
  

p = person("dcds", 22, "svds")
p.display()
c = person("default_name", 45, "none")
c.update("new_name", 25, "english")
