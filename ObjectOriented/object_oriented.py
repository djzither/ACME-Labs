# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Derek Robinson>
<Class>
<9/27/2025>
"""

from math import sqrt

class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color: (str): The color of the backpack
            max_size: the biggest size of backpack you can have
        """
        ##atributes of the backpack class
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents."""
        #putting stuff in the backpack
        if len(self.contents) >= self.max_size:
            print("No Room!")
        else:
            self.contents.append(item)

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        #remove
        self.contents.remove(item)

    def dump(self):
        """Empty the contents of the backpack."""
        #dump
        self.contents = []

    # Magic Methods -----------------------------------------------------------

    def __eq__(self, other):
        #sees if packs are equal
        if not isinstance(other, Backpack):
            return False
        return(self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents))

    def __str__(self):
        return (f"Owner: {self.name}\n"
                f"Color: {self.color}\n"
                f"Size: {len(self.contents)}\n"
                f"Max Size: {self.max_size}\n"
                f"Contents: {self.contents}")
    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

def test_backpack():

    test_pack = []
    test_pack = Backpack("Derek","black", 5)

    if test_pack.name != "Derek":
        print("Backpack.name assigned not correctly")

    for item in ["pencil", "pen", "paper", "computer"]:
        test_pack.put(item)

    print("contents: ", test_pack.contents)

    if len(test_pack.contents) > 5:
        print("Backpack.put works incorrectly")

    test_pack.take("pencil")

    for i in test_pack.contents:
        if i == "pencil":
            print("Backpack.take doesn't actually take")
    print("new contents after take", test_pack.contents)

    test_pack.dump()
    if test_pack.contents != []:
        print("Backpack.dump doesn't work correclty")
    print("new contents after dump", test_pack.contents)

# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)

# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    
    def __init__(self, name, color, max_size = 2, fuel = 10):
        """__init__  makes a jetpack with a name, color and fuel, and then it assigns with the old 
        backpack class """
        #over rides backpack and makes it better for jetpack
        super().__init__(name, color, max_size)
        self.fuel = fuel
    
    def fly(self, fuel_burn):
        """ fly makes the fuel go down and sees if there is enough fuel to burn"""
        ## sees how much fuel you burn and if enough
        if(self.fuel >= fuel_burn):
            self.fuel -= fuel_burn
        else:
            print("Not enough fuel!")

    def dump(self):
        """
        Empty the jetpack completely.

        This method overrides Backpack.dump() so that both the contents
        and the fuel tank are emptied.
        """
        #dumps and overides
        super().dump()
        self.fuel = 0

class ComplexNumber():
    def __init__(self, real, imag):
        """makes the instantiation"""
        #instantiate
        self.real = real
        self.imag = imag
    
    def conjugate(self):
        """makes the conjugate"""
        #make conjugate
        return ComplexNumber(self.real, -self.imag)

    def __str__(self):
        """this is going to print it when called"""
        #uses f strings to format
        if self.imag >= 0:
            return f"({self.real}+{self.imag}j)"
        else:
            return f"({self.real}-{abs(self.imag)}j)"
    
    def __abs__(self):
        """gets the abs val"""
        #uses the magnitude and sqr root formulas
        magnitude = sqrt((self.real**2) + (self.imag**2))
        return magnitude
    
    def __eq__(self, other):
        """Sees if equal"""
        ## returns true or false
        return self.real == other.real and self.imag == other.imag
            
    def __add__(self, other):
        """adds complex numbers"""
        #uses complex number forumula
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        """subtracts complex nums"""
        #uses complex sub formula
        return ComplexNumber(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        """multiplies complex nums"""
        # uses multiplication formula that is harder
        return ComplexNumber(self.real*other.real - self.imag*other.imag,
                             self.real*other.imag + self.imag*other.real)

    def __truediv__(self, other):
        """divides formula for complex"""
        #this one is a little longer
        den = other.real**2 + other.imag**2
        num_real = self.real*other.real + self.imag*other.imag
        num_imag = self.imag*other.real - self.real*other.imag
        return ComplexNumber(num_real/den, num_imag/den)

def test_ComplexNumber(a, b):
    """
    Compare our ComplexNumber class to Python's built-in complex type.
    Runs a set of operations and checks results match exactly.
    """
    py_cnum1 = complex(a, b)
    py_cnum2 = complex(a, b)

    my_cnum1 = ComplexNumber(a, b)
    my_cnum2 = ComplexNumber(a, b)

    # Conjugate
    if my_cnum1.conjugate().real != py_cnum1.conjugate().real or \
       my_cnum1.conjugate().imag != py_cnum1.conjugate().imag:
        print("conjugate() failed")

    # String
    if str(my_cnum1) != str(py_cnum1):
        print("__str__() failed")

    # Absolute value
    if abs(my_cnum1) != abs(py_cnum1):
        print("__abs__() failed")

    # Addition
    if complex(my_cnum1.real, my_cnum1.imag) + complex(my_cnum2.real, my_cnum2.imag) != \
       complex((my_cnum1 + my_cnum2).real, (my_cnum1 + my_cnum2).imag):
        print("__add__() failed")

    # Subtraction
    if complex(my_cnum1.real, my_cnum1.imag) - complex(my_cnum2.real, my_cnum2.imag) != \
       complex((my_cnum1 - my_cnum2).real, (my_cnum1 - my_cnum2).imag):
        print("__sub__() failed")

    # Multiplication
    if complex(my_cnum1.real, my_cnum1.imag) * complex(my_cnum2.real, my_cnum2.imag) != \
       complex((my_cnum1 * my_cnum2).real, (my_cnum1 * my_cnum2).imag):
        print("__mul__() failed")

    # Division
    if complex(my_cnum1.real, my_cnum1.imag) / complex(my_cnum2.real, my_cnum2.imag) != \
       complex((my_cnum1 / my_cnum2).real, (my_cnum1 / my_cnum2).imag):
        print("__truediv__() failed")

    print("All ComplexNumber tests passed!")



if __name__ == "__main__":
    test_backpack()
    test_ComplexNumber(3, 4)
    test_ComplexNumber(1, -2)
