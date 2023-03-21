import sys

from crossword import *
from crossword import Variable

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.crossword.variables:
            for w in self.crossword.words:
                if len(w) != var.length:
                    self.domains[var].remove(w)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps[x, y]
        any_revision = False

        if not overlap:
            return False
        else:
            (i, j) = overlap
            l = self.domains[x].copy()
            for wx in l:
                # Assuming initially that wx will not be arc consistent wrt y
                remove_from_domain = True
                for wy in self.domains[y]:
                    if wx[i] == wy[j]:
                        # arc consistency exists so should not remove wx from domain
                        remove_from_domain = False
                if remove_from_domain:
                    any_revision = True
                    self.domains[x].remove(wx)

        return any_revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        if not arcs:
            arcs = []
            for var1 in self.crossword.variables:
                for var2 in self.crossword.variables:
                    if var1 == var2:
                        continue
                    else:
                        arcs.append(tuple((var1, var2)))

        while len(arcs) > 0:
            (x,y) = arcs.pop()
            if self.revise(x, y):
                for n in self.crossword.neighbors(x):
                    arcs.append(tuple((n, x)))

        for var in self.crossword.variables:
            if not self.domains[var]:
                return False

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) == len(self.crossword.variables):
            return True
        else:
            return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # Check if all assignment values are distinct
        if len(list(assignment.values())) == len(set(list(assignment.values()))):

            # Check if values are of correct length
            for a in assignment:
                if a.length != len(assignment[a]):
                    return False

            # Check if there are any conflicts with the neigbour
            for x in assignment:
                for y in self.crossword.neighbors(x):
                    (i, j) = self.crossword.overlaps[x, y]

                    try:
                        assignment[y]
                    except:
                        continue
                    else:
                        if not assignment[x] == assignment[y]:
                            if not assignment[x][i] == assignment[y][j]:
                                return False
            return True

        return False

    def find_num_rule_out_neighbour(self, domain_word, var, assignment):

        num_rule_out_neighbour = 0

        for n in self.crossword.neighbors(var):
            if n in assignment:
                continue

            (i, j) = self.crossword.overlaps[var, n]

            for domain_neigbour in self.domains[n]:
                if not domain_word == domain_neigbour[j]:
                    num_rule_out_neighbour += 1

        return num_rule_out_neighbour

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        list_of_domain = list(self.domains[var])
        list_of_domain.sort(key=lambda list_of_domain: self.find_num_rule_out_neighbour(list_of_domain, var, assignment))

        return list_of_domain


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Dafualt the Max Domain variable to negative infinity.
        min_domain = float('inf')
        # Find out the minimum number of domain across all variables
        for v in self.crossword.variables:

            if v in assignment:
                continue
            l = len(self.domains[v])
            if l < min_domain:
                min_domain = l

        # Append all the variable which have minimum domains to a new list

        list_of_variable = []
        #list_of_variable.append(v for v in self.crossword.variables if len(self.domains[v]) == min_domain)
        for v in self.crossword.variables:
            if len(self.domains[v]) == min_domain and v not in assignment:

                list_of_variable.append(v)

        # Sort this list basis on the number of neighbours it has
        list_of_variable.sort(key=lambda list_of_variable : len(self.crossword.neighbors(list_of_variable)), reverse=True)

        return list_of_variable[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # Check if all variables have been assigned, return assignment

        if len(assignment) == len(self.crossword.variables):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for word in self.order_domain_values(var, assignment):
            assignment_new = assignment.copy()
            assignment_new[var] = word

            # check if the current assignment is consistent
            if self.consistent(assignment_new):
                result = self.backtrack(assignment_new)
                if result is not None:
                    return result

        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
