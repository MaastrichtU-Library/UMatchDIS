from DeepImageSearch import Load_Data,Search_Setup

import os
import shutil
from datetime import datetime
from tqdm import tqdm
import pandas as pd


input_needle_folder_name = "sample-data-200-100/thumbs"
input_haystack_folder_name = "sample-data-200-100/highres"
# folder_name = "sample-data2"
output_folder_name = "output/results"
output_pairs = True  # Set to True if you want to create subfolders for each pair of images

# Create output folder if it doesn't exist
ts = datetime.now().strftime("%Y%m%d_%H-%M-%S")
target_folder = output_folder_name + "_" + ts
os.makedirs(target_folder, exist_ok=True)



# Load images from a folder
needle_image_list = Load_Data().from_folder([input_needle_folder_name])
haystack_image_list = Load_Data().from_folder([input_haystack_folder_name])

print("Total Reference images count:",len(needle_image_list))
print("Total Haystack images count:",len(haystack_image_list))

# Set up the search engine
st = Search_Setup(image_list=needle_image_list,model_name='vgg19',pretrained=True,image_count=100)

# Index the reference (needle) images
st.run_index()

# Add to-be-matched (haystack) images to the index
st.add_images_to_index(haystack_image_list)

# Update metadata
metadata = st.get_image_metadata_file()
#metadata

# Create empty data frame and set colnames
df = pd.DataFrame(
    columns=[
        'thumbnail',
        'thumbnail_file',
        'matched_high-res'
    ]
)

# Plot similar images
for i in tqdm(range(0,len(needle_image_list)), desc="Finding similar images"):
    thumb_file_name = os.path.splitext(os.path.basename(needle_image_list[i]))[0]  # without extension
    thumb_file = os.path.basename(needle_image_list[i])                            # with extension
    thumb_file_path = needle_image_list[i]

    # Find 2 similar images and take the last one (the first is always the original thumbnail, because: perfect match))
    similar_img_path = list(st.get_similar_images(image_path=needle_image_list[i], number_of_images=2).values())[-1]

    # Code to plot the similar images
    #st.plot_similar_images(image_path=image_list[i], number_of_images=2)

    # Copy the files to the target folder
    if output_pairs:
        final_folder = os.path.join(target_folder, thumb_file_name)
        os.makedirs(final_folder, exist_ok=True)
    else:
        final_folder = target_folder

    shutil.copy(thumb_file_path, final_folder)
    shutil.copy(similar_img_path, final_folder)

    # Transform data into 3 columns
    new_row = pd.DataFrame({
        "thumbnail": [thumb_file_name],
        "thumbnail_file": [thumb_file],
        "matched_high-res": [os.path.basename(similar_img_path)]
    })

    # Concatenate new row to data frame
    df = pd.concat([df, new_row])


# use thumbnail as index. Allows accessing rows by thumbnail name (e.g. `df.loc["1070"]` )
df.set_index('thumbnail', inplace=True)
df.reset_index(names="FileNumber", inplace=True)    # Changes the name of the index column

# Write to CSV file
df.to_csv(f"{target_folder}/overview_matched_images.csv", index=False)  # Write to csv because of VLOOKUP issues on FileNumber column in .xlsx