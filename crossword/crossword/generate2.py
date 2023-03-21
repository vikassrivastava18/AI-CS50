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
        for var in self.domains.keys():
            variable_domain = set()

            for word in self.domains[var]:
                if len(word) == var.length:
                    variable_domain.add(word)
            self.domains[var] = variable_domain

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revision = False
        overlap = self.crossword.overlaps[(x,y)]
        domain_arc_consistent = set()

        if overlap:
            for word in self.domains[x]:
                arc_consistent = False
                for word2 in self.domains[y]:
                    if word[overlap[0]] == word2[overlap[1]]:
                        arc_consistent = True
                        domain_arc_consistent.add(word)
                        break
                if not arc_consistent:
                    revision = True
            self.domains[x] = domain_arc_consistent
        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        ques = set()

        for domain in self.domains:
            neighbors = self.crossword.neighbors(domain)
            for neighbor in neighbors:
                if (domain, neighbor) not in ques or (neighbor, domain) not in ques:
                    ques.add((domain, neighbor))

        while len(ques) > 0:
            (x, y) = ques.pop()
            if not self.revise(x,y):
                continue
            else:
                neighbors = self.crossword.neighbors(x).remove(y)
                if neighbors:
                    for neighbor in neighbors:
                        if (x, neighbor) not in ques or (neighbor, x) not in ques:
                            ques.add((x, neighbor))

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if not len(assignment.keys()) == len(self.domains.keys()):
            return False
        for var in assignment.keys():
            if len(assignment[var]) == 0:
                return False

        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        unique_words = []
        for var in assignment.keys():
            if not len(assignment[var]) == var.length:
                return False
            if assignment[var] in unique_words:
                return False
            else:
                unique_words.append(assignment[var])

            neighbors = self.crossword.neighbors(var)
            for neighbor in neighbors:
                try:
                    overlap = self.crossword.overlaps[(var, neighbor)]
                    if not assignment[var][overlap[0]] == assignment[neighbor][overlap[1]]:
                        return False
                except Exception as e:
                    continue
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        neighbors = self.crossword.neighbors(var)
        max_eliminate = 0
        eliminate_list = []

        for domain in self.domains[var]:
            eliminated = 0
            for neighbor in neighbors:
                overlap = self.crossword.overlaps[(var, neighbor)]
                eliminated += len([n_dom for n_dom in self.domains[neighbor] if not domain[overlap[0]] == n_dom[overlap[1]]])
            eliminate_list.append({'domain':domain, 'eliminated': eliminated})

        eliminate_list = sorted(eliminate_list, key = lambda i: i['eliminated'],reverse=True)
        return [key['domain'] for key in eliminate_list]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        variable_list = []

        for var in self.domains.keys():
            if var not in assignment:
                variable_list.append({'variable': var, 'count': len(self.domains[var])})

        variable_list = sorted(variable_list, key=lambda i: i['count'])

        return variable_list[0]['variable']

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if len(assignment) == len(self.domains.keys()):
            return assignment

        # Try a new variable
        var = self.select_unassigned_variable(assignment)
        print("variable", var)
        for value in self.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
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
