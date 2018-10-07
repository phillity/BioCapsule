import os

database = "images//caltech"
for dir in os.listdir(database):
    i = 1
    for file_name in os.listdir(database + "//" + dir):
        new_file_name = str(i)
        while len(new_file_name) < 3:
            new_file_name = "0" + new_file_name
        new_file_name = dir + "_" + new_file_name + ".jpg"

        file_path = database + "//" + dir + "//" + file_name
        new_file_path = database + "//" + dir + "//" + new_file_name

        print(file_path)
        print(new_file_path)
        os.rename(file_path,new_file_path)

        i = i + 1
