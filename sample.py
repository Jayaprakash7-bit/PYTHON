def display_menu():
    print("\n---- To-Do List Menu ----")
    print("1. Add Task")
    print("2. View Tasks")
    print("3. Remove Task")
    print("4. Exit")

def main():
    tasks = []

    while True:
        display_menu()
        choice = input("Choose an option (1-4): ")

        if choice == '1':
            task = input("Enter the task: ")
            tasks.append(task)
            print("Task added.")

        elif choice == '2':
            if not tasks:
                print("Your to-do list is empty.")
            else:
                print("\nYour Tasks:")
                for i, task in enumerate(tasks, start=1):
                    print(f"{i}. {task}")

        elif choice == '3':
            if not tasks:
                print("No tasks to remove.")
            else:
                task_no = int(input("Enter the task number to remove: "))
                if 1 <= task_no <= len(tasks):
                    removed = tasks.pop(task_no - 1)
                    print(f"Removed task: {removed}")
                else:
                    print("Invalid task number.")

        elif choice == '4':
            print("Exiting To-Do List App. Bye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()