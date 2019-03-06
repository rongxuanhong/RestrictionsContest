NORMAL_PATH = "data/first_round_data/jinnan2_round1_train_20190222/normal"

RESTRICTED_PATH = "data/first_round_data/jinnan2_round1_train_20190222/restricted"

TEST_PATH = "data/first_round_data/jinnan2_round1_test_a_20190222"

TRAIN_NO_POLY_JSON_PATH = "data/first_round_data/jinnan2_round1_train_20190222/train_no_poly.json"

SUBMIT_PATH = 'submit'

BEST_MODEL_PATH = 'code/best_model'

labels = ["铁壳打火机", "黑钉打火机", "刀具", "电池电容", "剪刀", ]

label_to_ind = {label: ind for ind, label in enumerate(labels)}
ind_to_label = {ind: label for ind, label in enumerate(labels)}

img_weight = 224
img_height = 224

seed = 2019

# batch_size = 8

gpus = '0'
