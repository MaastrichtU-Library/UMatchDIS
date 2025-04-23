# input_needle_folder_name = "/home/maarten/work/git_workdir/fotocollectie/thumbs_opgeschoond"
# input_haystack_folder_name = "/media/maarten/WD_BLACK_MC/um_fotocollectie/highres"
input_needle_folder_name = "input/sample-data-15-110/thumbs"
input_haystack_folder_name = "input/sample-data-15-110/highres"
# input_needle_folder_name = "sample-data-5-100/thumbs"
# input_haystack_folder_name = "sample-data-5-100/highres"

output_folder_name = "output/"
output_pairs = True  # Set to True if you want to create subfolders for each pair of images

similarity_score_threshold = 0.25   # Set a matching threshold. Photo matches with a lower score will be ignored

# model_name = "vgg19"
# model_name = "caformer_b36.sail_in22k_ft_in1k_384"
# model_name = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
model_name = "regnety_1280.swag_ft_in1k"
