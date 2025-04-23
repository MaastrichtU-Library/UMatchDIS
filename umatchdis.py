from DeepImageSearch import Load_Data,Search_Setup

import os
import shutil
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Use vars from config file
import uconfig as cfg

def main():
    # Create output folder if it doesn't exist
    ts = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    target_folder = cfg.output_folder_name + "_" + ts + "_" + cfg.model_name
    os.makedirs(target_folder, exist_ok=True)


    # Load images from a folder
    needle_image_list = Load_Data().from_folder([cfg.input_needle_folder_name])
    haystack_image_list = Load_Data().from_folder([cfg.input_haystack_folder_name])
    all_images_list = Load_Data().from_folder([cfg.input_needle_folder_name, cfg.input_haystack_folder_name])

    print("Total Needle images count:  ",len(needle_image_list))
    print("Total Haystack images count:",len(haystack_image_list))

    # Set up the search engine
    #st = Search_Setup(image_list=needle_image_list, model_name=cfg.model_name, pretrained=True, image_count=100)
    st = Search_Setup(image_list=needle_image_list, model_name=cfg.model_name, pretrained=True)
    # st = Search_Setup(image_list=all_images_list, model_name=cfg.model_name, pretrained=True)


    # Index the reference (needle) images (i.e. Extract features from images and indexes them)
    st.run_index()

    # Add to-be-matched (haystack) images to the index (i.e. appends the feature vectors of the new images to the index)
    st.add_images_to_index(haystack_image_list)

    # Update metadata
    # metadata = st.get_image_metadata_file()
    # metadata

    # Create empty data frame and set colnames
    df = pd.DataFrame(
        columns=[
            'thumbnail',
            'thumbnail_file',
            'matched_high-res',
            'match_score'
        ]
    )

    # Plot similar images
    for i in tqdm(range(0,len(needle_image_list)), desc="Finding similar images"):
        thumb_file_name = os.path.splitext(os.path.basename(needle_image_list[i]))[0]  # without extension
        thumb_file = os.path.basename(needle_image_list[i])                            # with extension
        thumb_file_path = needle_image_list[i]

        # Find 5 similar images (ranked from high to low score), loop over the matches and take the highres image with the highest score.
        # Note that the first match is always the original image, i.e. the 'needle'-image, as that is a perfect match (score = 1)
        results = st.get_similar_images(image_path=needle_image_list[i], number_of_images=5)

        similar_img_path = ""
        similar_img_score = 0
        for match in results:
            #print(f"Image: {match['image_path']}, Similarity Score: {match['score']}")
            if "highres/" not in match['image_path']:
                similar_img_path = "no match"
                similar_img_score = 0
            else:
                if match['score'] < cfg.similarity_score_threshold:
                    similar_img_path = "low confidence match"
                    similar_img_score = match['score']
                else:
                    similar_img_path = match['image_path']
                    similar_img_score = match['score']
                    break

        # Code to plot the similar images
        #st.plot_similar_images(image_path=image_list[i], number_of_images=2)

        if similar_img_path != "no match" and similar_img_path != "low confidence match":
            # Copy the files to the target folder
            if cfg.output_pairs:
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
            "matched_high-res": [os.path.basename(similar_img_path)],
            "match_score": [similar_img_score]
        })

        # Concatenate new row to data frame
        df = pd.concat([df, new_row])


    # use thumbnail as index. Allows accessing rows by thumbnail name (e.g. `df.loc["1070"]` )
    df.set_index('thumbnail', inplace=True)
    df.reset_index(names="FileNumber", inplace=True)    # Changes the name of the index column

    # Write to CSV file
    df.to_csv(f"{target_folder}/overview_matched_images.csv", index=False)  # Write to csv because of VLOOKUP issues on FileNumber column in .xlsx


if __name__ == "__main__":
    main()