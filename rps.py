import random
player = input("Enter A any of this options  rock, paper,scissors : ")
computer = random.choice(["rock", "paper", "scissors"])
print("player = "+ player)
print("computer = "+computer)
playagain = True
while playagain:
    if player == computer:
        print("It's a tie!")
    elif player == "rock" and computer == "scissors":
            print("Player wins!")
    elif player == "paper" and computer == "rock":
            print("Player wins!")
    elif player == "scissors" and computer == "paper":
            print("Player wins!")
    else:
            print("Computer wins😊🎉!")
    playagain = input("do want to play y for yes and n for no: ")
    if playagain.lower() == "y":
        player = input("Enter A any of this options  rock, paper,scissors : ")
        print("player = "+ player)
        print("computer = "+computer)
        continue

    else:
          
          print("thank you for playing!")
          playagain = False   
          print("congrats🎉🎉")