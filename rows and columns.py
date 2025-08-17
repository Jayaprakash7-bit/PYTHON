
M = int(input("Enter number of rows (M): "))
N = int(input("Enter number of columns (N): "))

grid = []

for i in range(M):
    row = []
    for j in range(N):
        row.append(f"({i},{j})")  # Each cell shows its position
    grid.append(row)

for row in grid:
    print("  ".join(row))
