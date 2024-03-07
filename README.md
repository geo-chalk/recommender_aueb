# Recommander Systems 

## Deploying Machine Learning Applications
Migrating from Jupyter Notebooks to Python Applications

Course Material for Recommender Systems  (INF499) - MSc Data Science

## Set up environments
### Step 1:

Manually create the environment and add dependencies as the project is being worked.
This is a good option to start building the application, as you won't know beforehand
which libraries you are going to use.
`conda create -n rec_sys_app python=3.11.0 ipython --yes `

`conda activate rec_sys_app`

### Step 2:
After finishing the project, create a conda environment, with the libraries you have used in the project:

Reference file:
```yaml
name: rec_sys_app
channels:
  - defaults
dependencies:
  - python=3.11.0
  - ipython=8.20.0
  - numpy=1.26.4
  - pandas=2.1.4
  - scikit-learn=1.2.2
  - pip=23.3.1
  - catboost=1.2
  - jupyter=1.0
  - jupyterlab=4.0.8
```

Use the following commands to create and activate:
```
conda env create -f setup/conda_env.yml
conda activate rec_sys_app
```

### Step 3:
Use docker (TBU)

## Set up folder structure
There are multiple ways to set up a python environment. Usually, the code 
is maintained under the `./src` folder. Inside the `src` folder, 
more folders (modules) should be places, each of which should have a common *'theme'*.

## Clean Code

For more info see: https://peps.python.org/pep-0008/

### Use OOP approach 
Try using Object Oriented Programming approaches, but using classes and functions.
Methods and Objects should have easy to understand names and executes small functions.
Moreover, any possible argument that is subject to change should be part of the method arguments.

### Docstrings
It is very important to make the code clear and easy to read. <br>
How many times have you tried to understand what your code does? 
This can be avoided using docstrings and comments. 

### Typehints
Python is a dynamically typed language, and a results, typehints are important.
They make sure that we are expecting the correct type, which is useful as our code 
becomes more and more complex.

Moreover, in modern editors, we can hover over a function and see the expected 
```Python
def my_example_func(foo: str) -> list[str]:
    return [foo for _ in range(10)]

incorrect_type: str = my_example_func("rec_sys")
```

### Example:
Try answering the following questions:
* What does the function below does? 
* What type should the input arguments be?
* What type will it return?
Hard to understand
```python
import math
def evaluate(x, n_terms):
    result = 0.0
    sign = 1
    for i in range(0, n_terms):
        term = (x**(2*i + 1)) / math.factorial(2*i + 1)
        if i % 2 == 1:
            sign *= -1
        result += sign * term
    return result
```

How about now?
```python
import math
def evaluate_polynomial(x: float, n_terms: int) -> float:
    """
    Evaluate a trigonometric expression using Taylor series approximation.
    
    The expression is evaluated using the Taylor series expansion of sin(x):
    sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + ...

    Parameters:
    x: The angle in radians.
    n_terms: The number of terms to use in the Taylor series approximation.

    Returns:
    float: The value of the trigonometric expression at the given angle.
    
    >>> evaluate_polynomial(math.pi/4, 10)
    0.7071067811865476
    """
    result = 0.0
    sign = 1
    for i in range(0, n_terms):
        term = (x**(2*i + 1)) / math.factorial(2*i + 1)
        if i % 2 == 1:
            sign *= -1
        result += sign * term
    return result
```

## Running the code
The repository should contain a `main.py` function under `src` which would execute the program
and generate the desired output.

Provide clear instruction so that code can be reproduced.
Example for this repo:

1. cd to project root
2. run `python src/main.py`