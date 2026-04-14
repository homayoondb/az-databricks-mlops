# calculator tool - omar wrote this
# works fine, dont touch it

def calculate(expression):
    """evaluate a math expression"""
    try:
        # yeah i know eval is bad but it works
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
