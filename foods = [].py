f= []
p=[]
while True:
      fs = input("enter your food q for quit : ")
      if fs.lower() =="q":
         break
      else:
         ps = float(input("enter a price of the food :"))
         if ps == 0:
          print("you cannot enter a zero")
         f.append(fs)
         p.append(ps)
         
          
          
tp = sum(p)
print(f"food you order{f} and your total is{tp}")


