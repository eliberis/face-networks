import pandas as pd
from os import mkdir
from shutil import copyfile
from tqdm import tqdm as print_progress

def main():
    # Partition the Adience dataset images into subfolders according to gender
    mkdir("data/processed")
    mkdir("data/processed/train")
    mkdir("data/processed/train/f")
    mkdir("data/processed/train/m")
    mkdir("data/processed/test")
    mkdir("data/processed/test/f")
    mkdir("data/processed/test/m")

    test_fold = 4

    for fold_num in range(5):
        descr_file = "data/fold_{0}_data.txt".format(fold_num)
        mode = "test" if fold_num == test_fold else "train"
        df = pd.read_csv(descr_file, sep="\t")

        for _, entry in print_progress(df.iterrows()):
            if entry["gender"] not in ['f', 'm']:
                continue

            entry = entry.to_dict()
            img_file = "data/faces/{user_id}/coarse_tilt_aligned_face." \
                       "{face_id}.{original_image}".format(**entry)
            dest_file = "data/processed/{mode}/{gender}" \
                        "/{original_image}".format(mode=mode, **entry)
            copyfile(img_file, dest_file)

if __name__ == "__main__":
    main()
