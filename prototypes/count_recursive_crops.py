import os
import glob

# DIR = "/home/quentin/Insync/qgeissmann@gmail.com/Google Drive - Shared with me/PIEE Lab - Shared/Projects/ALAN/pitfall_classification/Arthropoda/Insecta/Coleoptera/"
# pattern = os.path.join(DIR, "**","*.png")
# print(pattern)
# print(len([f for f in glob.glob(pattern, recursive=True)]))


DIR = "/home/quentin/Insync/qgeissmann@gmail.com/Google Drive - Shared with me/PIEE Lab - Shared/Projects/ALAN/pitfall_classification/Arthropoda"
DIR = "/home/quentin/Desktop/pitfall_classif/Arthropoda"

for root, subdirs, files in os.walk(DIR):
    # print(root)
    # print([f for f in glob.glob(os.path.join(root, "**", "*.png"))])
    n_files_subdirs = len([f for f in glob.glob(os.path.join(root, "**", "*.png"), recursive=True)])
    taxa = os.path.relpath(root, start=DIR)
    print(taxa,",", n_files_subdirs)


