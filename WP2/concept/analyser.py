#!/usr/bin/python3

import ast


class CallNameExtractor(ast.NodeVisitor):
    """(Shallow) extracts the name of a function call."""

    def visit_Name(self, node):  # noqa: N802
        self.name = node.id


class CallAnalyser:
    """Traverses a source tree and extracts function calls."""

    def __init__(self):
        """Constructs an empty call analyser report."""
        self.stats = []

    def get_func_calls(self, tree):
        """Traverses source tree and extracts function calls."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # print(ast.dump(node))
                stat = {"kwargs": {}}
                stat["lineno"] = node.func.lineno
                stat["colno"] = node.func.col_offset
                extractor = CallNameExtractor()
                extractor.visit(node)
                stat["name"] = extractor.name
                for kwarg in node.keywords:
                    if isinstance(kwarg.value, ast.Constant):
                        name = kwarg.arg
                        value = kwarg.value.value
                        stat["kwargs"][name] = value
                self.stats.append(stat)

    def report(self):
        """Print call analysis."""
        for call in self.stats:
            lineno = call["lineno"]
            colno = call["colno"]
            name = call["name"]
            if name == "DecisionTreeClassifier":
                min_samples = call["kwargs"]["min_samples_leaf"]
                if min_samples < 5:
                    print(
                        f"warning: line {lineno}:{colno} DecisionTreeClassifier"
                        f" initialised with min_samples_leaf = {min_samples}"
                    )


def main():
    """Main method: reads source code and calls analyser."""
    with open("test_code.py", "r") as source:
        tree = ast.parse(source.read())
    #    print(ast.dump(tree))
    analyser = CallAnalyser()
    analyser.get_func_calls(tree)
    analyser.report()


if __name__ == "__main__":
    main()
