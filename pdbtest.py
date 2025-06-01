import pdb

def multiply(a, b):
    result = a * b
    pdb.set_trace()  # This will pause execution and open the debugger
    return result

def main():
    x = 5
    y = 3
    print(f"The product of {x} and {y} is {multiply(x, y)}")

if __name__ == "__main__":
    main()
