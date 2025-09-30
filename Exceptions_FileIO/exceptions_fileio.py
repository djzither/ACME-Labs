# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>
<Class>
<Date>
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    #lets get those nums
    step_1 = input("Enter a 3-digit number where the first and last "
                                  "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("Must be a 3 digit number")

    #check to see if first and last digits differ by desired amount
    digits = [int(i) for i in str(step_1)]
    if abs(digits[2] - digits[0]) < 2:
        raise ValueError("First and last digit must differ by a value of 2 or more")
    #input reversed and reverse
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    reversed_digits = digits[::-1]
    first_reversed_int = int("".join(str(i) for i in reversed_digits))
    #see if they did it right
    if str(first_reversed_int) != step_2:
        raise ValueError("The reverse of the first number and the reverse you entered are not the same!")
    step_3 = input("Enter the positive difference of these numbers: ")
    #see if they did the right differece
    if step_3 != str(abs(int(step_2) - int(step_1))):
        raise ValueError("The difference you entered is not the right answer")

    step_4 = input("Enter the reverse of the previous result: ")
    reversed_difference = step_3[::-1]
    #see if they reversed it correctly
    if step_4 != reversed_difference:
        raise ValueError("The reverse of the difference you entered is not correct")

    final = int(step_3) + int(step_4)
    
    if final != 1089:
        raise ValueError("you didn't add in the final two together correctly ")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")

# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    #create the walk
    walk = 0
    directions = [1, -1]
    try: 
        for _ in range(int(max_iters)):
            walk += choice(directions)
    #we are going to check the keyboard inturrupt
    except KeyboardInterrupt:
        print(f"process inturrupted at iteration {_}")
        return walk
    else:
        print("Walk completed")
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
    # Problem 3
    def __init__(self, filename):
        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        #we will read in the file and make sure it is correct
        while True:
            filename = input("Enter a file name: ")
            try:
                with open(filename, "r") as f:
                    self.contents = f.read()
                    self.filename = filename
                break
            
            except (FileNotFoundError, TypeError, OSError, UnicodeDecodeError):
                filename = input("please enter a valid file name: ")
            

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """
        #these are the possible python file read types
        valid_nodes = {"r", "w", "x", "a"}
        if mode not in valid_nodes:
            raise ValueError(f"invalid mode {mode}, has to be on of {valid_nodes}")


    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include a
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """
        #sees what case to write it in
        if case not in {"upper", "lower"}:
            raise ValueError("case must be 'upper' or 'lower'")
        self.check_mode(mode)
        text = self.contents.upper() if case == "upper" else self.contents.lower()
        with open(outfile, mode) as f:
            f.write(text)

    def reverse(self, outfile, mode='w', unit='line'):
        """ Write the data to the outfile in reverse order. Include a
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """
        #check the unit we are reversing by
        if unit not in {"line", "work"}:
            raise ValueError("unit has to be 'line' or 'word'")
        self.check_mode(mode)
        lines = self.contents.splitlines()
        #reverse it by the unit
        if unit == "word":
            text = "\n".join(" ".join(line.split()[::-1]) for line in lines)
        else:
            text = "\n".join(lines[::-1])
        #write it back
        with open(outfile, mode) as f:
            f.write(text)


    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """
        self.check_mode(mode)
        #split up the lines to transpose
        lines = [line.split() for line in self.contents.splitlines]
        if not lines:
            transposed = []
        else:
            transposed = zip(*lines)
            text = "\n".join(" ".join(word for word in row) for row in transposed)
        #write it to file
        with open(outfile, mode) as f:
            f.write(text)
        

    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """
        #gonna do all the actions we need
        total_chars = len(self.contents)
        letters = sum(c.isalpha() for c in self.contents)
        digits = sum(c.isdigit() for c in self.contents)
        whitespace = sum(c.isspace() for c in self.contents)
        num_lines = len(self.contents.splitlines())
        #this gives the output for string
        return(f"Source file: {self.filename}\n"
        f"Total Characters: {total_chars}\n"
        f"Alphabetic characters {letters}\n"
        f"Numerical characters {digits}\n"
        f"Whitespace chacarters {whitespace}\n"
        f"number of lines:{num_lines}")
        
    
    if __name__ == "__main__":
        arithmagic()