import os

for filename in os.listdir('booksim2/runfiles/mesh'):
    f = os.path.join('booksim2/runfiles/mesh', filename)
    if os.path.isfile(f):
        with open(f, 'r') as file:
            data = file.readlines()
        cwd = os.getcwd()
        index = data[5].index('=')
        first_part = data[5][:index + 2]
        last_part = data[5][index + 2:]
        if not last_part.startswith(cwd):
            data[5] = first_part + cwd + last_part
        with open(f, 'w') as file:
            file.writelines(data)