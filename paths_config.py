import os

location_dict = dict([])

if "SHARED" in os.listdir("/mnt/"):
    shared = "/mnt/SHARED/"
    local = "/scratch/"
    project_folder = "/mnt/SHARED/ameinke03/projects/fair_face/"
else:
    shared = "/mnt/qb/hein/"
    local = "/home/hein/ameinke03/"
    project_folder = "/mnt/qb/hein/ameinke03/projects/fair_face/"


# Define the location of your celebA directory here
location_dict["CelebA"] = local + "datasets/celebA/"
