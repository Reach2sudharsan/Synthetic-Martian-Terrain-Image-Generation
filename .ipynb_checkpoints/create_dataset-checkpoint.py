import os
import shutil
import random

def retrieve_images_and_store(dataset_path, outdir, filename_prefix):
    folders = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    os.mkdir(outdir)
    os.mkdir(f"{outdir}/images")
    os.mkdir(f"{outdir}/labels")

    i = 0
    for folder in folders:
        image_folder_path = os.path.join(dataset_path, f"{folder}/images")
        labels_folder_path = os.path.join(dataset_path, f"{folder}/labels")

        files = os.listdir(image_folder_path)

        for file in files:
            image_src_path = os.path.join(image_folder_path, file)
            filename, extension = os.path.splitext(file)
            label_src_path = os.path.join(labels_folder_path, f"{filename}.txt")

            image_dest_path = os.path.join(f"{outdir}/images", f"{filename_prefix}_{i+1}{extension}")
            label_dest_path = os.path.join(f"{outdir}/labels", f"{filename_prefix}_{i+1}.txt")

            shutil.copy(image_src_path, image_dest_path)
            shutil.copy(label_src_path, label_dest_path)
            i += 1

    print(f"Retrieved images from\n {dataset_path}\n and stored in\n {outdir}")

def create_dataset(real_path, synthetic_path, size, ratio, tvt_split="90-5-5"):

    # real_files = [os.path.join(real_path, file) for file in os.listdir(real_path)]
    # synthetic_files = [os.path.join(synthetic_path, file) for file in os.listdir(synthetic_path)]
    real_images = os.listdir(os.path.join(real_path, "images"))
    real_labels = os.listdir(os.path.join(real_path, "labels"))

    synthetic_images = os.listdir(os.path.join(synthetic_path, "images"))
    synthetic_labels = os.listdir(os.path.join(synthetic_path, "labels"))

    tvt = tvt_split.split("-")
    train_prp = int(tvt[0])/100
    valid_prp = int(tvt[1])/100

    train_size = round(train_prp*size)
    real_train_size = round((1-ratio)*train_size)
    syn_train_size = train_size - real_train_size
    valid_size = round(valid_prp*size)
    test_size = size - train_size - valid_size

    parent_outdir = "yolo_datasets"
    os.makedirs(parent_outdir, exist_ok=True)
    
    outdir = os.path.join(parent_outdir, f"syn{ratio}_size{size}_{tvt_split}")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    os.mkdir(f"{outdir}/train")
    os.mkdir(f"{outdir}/valid")
    os.mkdir(f"{outdir}/test")

    # print("OKOK")
    # print(len(real_images), real_train_size)
    real_train_images = random.sample(real_images, real_train_size)
    real_train_labels = [os.path.splitext(file)[0]+".txt" for file in real_train_images]

    syn_train_images = random.sample(synthetic_images, syn_train_size)
    syn_train_labels = [os.path.splitext(file)[0]+".txt" for file in syn_train_images]

    remaining_files = list(set(real_images) - set(real_train_images))
    valid_images = random.sample(remaining_files, valid_size)
    valid_labels = [os.path.splitext(file)[0]+".txt" for file in valid_images]

    remaining_files = list(set(remaining_files) - set(valid_images))
    test_images = random.sample(remaining_files, test_size)
    test_labels = [os.path.splitext(file)[0]+".txt" for file in test_images]

    def copy_files(files, source_dir, target_dir, remake=True):
        if remake:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.mkdir(target_dir)
            
        for f in files: 
            src_path = os.path.join(source_dir, f)
            dest_path = os.path.join(target_dir, f)
            shutil.copy(src_path, dest_path)
    
    copy_files(real_train_images, os.path.join(real_path, "images"), f"{outdir}/train/images")
    copy_files(real_train_labels, os.path.join(real_path, "labels"), f"{outdir}/train/labels")

    copy_files(syn_train_images, os.path.join(synthetic_path, "images"), f"{outdir}/train/images", False)
    copy_files(syn_train_labels, os.path.join(synthetic_path, "labels"), f"{outdir}/train/labels", False)

    copy_files(valid_images, os.path.join(real_path, "images"), f"{outdir}/valid/images")
    copy_files(valid_labels, os.path.join(real_path, "labels"), f"{outdir}/valid/labels")

    copy_files(test_images, os.path.join(real_path, "images"), f"{outdir}/test/images")
    copy_files(test_labels, os.path.join(real_path, "labels"), f"{outdir}/test/labels")

    with open(os.path.join(outdir, "data.yaml"), "w") as f:
        f.write(
            "train: ../train/images\n"
            "val: ../valid/images\n"
            "test: ../test/images\n"
            "\n"
            "nc: 1\n"
            "names: ['crater']\n"
        )

    print(f"Created {outdir} dataset")

if __name__ == "__main__":
    parent_outdir = "crater_datasets"
    if os.path.exists(parent_outdir):
        shutil.rmtree(parent_outdir)
    os.mkdir(parent_outdir)

    # Real Craters
    real_craters_path = "datasets/RealCraters"
    real_craters_outdir = "crater_datasets/RealCraterImages"
    filename_prefix = "real_crater"
    retrieve_images_and_store(real_craters_path, real_craters_outdir, filename_prefix)

    # Synthetic Craters - Stable Diffusion
    stbl_diff_path = "datasets/StblDiffCraters"
    stbl_diff_outdir = "crater_datasets/StblDiffCraterImages"
    filename_prefix = "stbl_diff_crater"
    retrieve_images_and_store(stbl_diff_path, stbl_diff_outdir, filename_prefix)

    # Synthetic Craters - ControlNet
    controlnet_path = "datasets/CtrlNetCraters"
    controlnet_outdir = "crater_datasets/CtrlNetCraterImages"
    filename_prefix = "controlnet_crater"
    retrieve_images_and_store(controlnet_path, controlnet_outdir, filename_prefix)

    # print(f"Real Count: {len(os.listdir(real_craters_outdir))}")
    # print(f"Sync Count (Stable Diffusion): {len(os.listdir(stbl_diff_outdir))}")
    # print(f"Sync Count (ControlNet): {len(os.listdir(controlnet_outdir))}")
    real_path = "crater_datasets/RealCraterImages"
    synthetic_path = "crater_datasets/StblDiffCraterImages"
    size = 1000

    ratio = 0.00
    create_dataset(real_path, synthetic_path, size, ratio, tvt_split="90-5-5")

    # ratio = 0.05
    # create_dataset(real_path, synthetic_path, size, ratio, tvt_split="90-5-5")

    # ratio = 0.50
    # create_dataset(real_path, synthetic_path, size, ratio, tvt_split="90-5-5")




