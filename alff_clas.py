"""Multimodal brain image classifier for MDD patient"""
import os
import errno
import sys
import traceback
import attr
import p_datasets
import tensorflow as tf
import numpy as np
import nibabel as nb
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn.image import load_img, new_img_like
#from nilearn import plotting
from sklearn.model_selection import train_test_split

sys.path.append("/home/foucault/projects/mdd_ml/source")
import loggy
try:
    LOGDIR = "/home/foucault/projects/mdd_ml/logs/"
    fileName = LOGDIR+os.path.basename(__file__)+".log"
    logger = loggy.logger(fileName, __name__)
except NameError:
    fileName = "/dev/null"
    logName = "jupyter"
    print("In Jupyter")
    logger = loggy.logger(fileName, logName)

DATAPATH = ["/home/foucault/data/mdd_data/"]
MODELITIES = ["MDD_ASL", "MDD_fMRI", "MDD_GQI", "MDD_VBM"]
GROUPS = ["01_HC", "02_D", "03_ID", "04_SH", "control", "depression",
          "ideation", "selfharm", "Depression", "healthy control"]
MEASURES = ["02mfALFF", "03mReGM", "02_mfALFF_depression", "03_mReHo_depression",
            "02_mfALFF_idea", "03_mReHo_idea", "02_mfALFF", "03_mReHo", "WM", "nqa0", "iso", "gfa"]


@attr.s
class DataPlaceholder(object):
    project_path = attr.ib(default=None)
    input_dirs = attr.ib(default=None)
    f_figure = attr.ib(default=None)
    d_results = attr.ib(default="/home/foucault/projects/results/")
    d_figures = attr.ib(default="/home/foucault/projects/figures/")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def slice_show(data, fname=None, n_slice=None):
    """Need to modify"""
    n = 0
    slice = 0
    if n_slice == None:
        n_slice = 1
    fig, ax = plt.subplots(1, n_slice, figsize=[18, 3])
    for _ in range(n_slice):
        #ax[n].imshow(data[:, :, slice], 'gray')
        ax.imshow(data[:, :], 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Slice number: {}'.format(slice), color='r')
        n += 1
        slice += 8

    fig.subplots_adjust(wspace=0, hspace=0)
    if fname:
        logger.info(f"Saving slice: {fname}")
        fig.savefig(fname)
    plt.close("all")


def examine_img(data):
    tmp_figure_file = "/home/foucault/projects/mdd_ml/figures/tmp/tmp.png"
    slice_show(data, fname=tmp_figure_file)


def check_img_dim(in_file, shape_tuple):
    img = load_img(in_file)
    img_array = img.get_fdata()
    exception = []
    if img_array.shape != shape_tuple:
        print("Caught exception: Image shape")
        print("file: "+in_file)
        result = in_file
    elif img_array.shape == shape_tuple:
        result = None
    return result


def check_multiple_img_dim(in_files, img_shape):
    check_dim_results = []
    for i, f in enumerate(in_files):
        result = check_img_dim(f.as_posix(), img_shape)
        check_dim_results.append(result)
    return check_dim_results


def gen_fig_fname(in_file):
    """ Generate file name for figure storage
    And generate corresponding directory """
    _path = in_file.parts[5:]
    p_path = Path.cwd().parent.joinpath("figures")
    for i in _path:
        p_path = p_path.joinpath(i)
    p_path = p_path.with_suffix(".png")
    d_path = p_path.parent
    if not d_path.exists():
        logger.info(f"mkdir: {d_path}")
        d_path.mkdir(parents=True, exist_ok=True)
    return p_path


def read_file_obsoleted():
    quary_pattern = "**/mfALFFMap*"
    assert_n_subjects = 62 # 02D
    target_path = DATAPATH[0]+MODELITIES[1]+"/"+GROUPS[1]+"/"+MEASURES[2]
    print(f"Parcing path: {target_path}")
    _p = Path(target_path)
    if not _p.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), _p)
    in_files = list(_p.glob(quary_pattern))

    # check number of subjects
    n_subjects = len(in_files) # n_subjects = 62
    assert n_subjects == assert_n_subjects

    # check image shape
    img_shape = (53, 63, 46)
    chk_results = check_multiple_img_dim(in_files, img_shape)


    imgs_array = []
    for i, f in enumerate(in_files):
        img = load_img(f.as_posix())
        img_array = img.get_fdata()
        img_array = np.rot90(img_array, 1)
        imgs_array.append(img_array)
        #fname = gen_fig_fname(f)
    return imgs_array


def read_imgs_old(target_path, quary_pattern, assert_n_subjects, assert_img_shape):
    logger.info(f"Parcing path: {target_path}")
    _p = Path(target_path)
    if not _p.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), _p)
    in_files = list(_p.glob(quary_pattern))

    # check number of subjects
    n_subjects = len(in_files)
    logger.info(f"Number of subject: {n_subjects}")
    if assert_n_subjects:
        assert n_subjects == assert_n_subjects

    # check image shape
    logger.info(f"Assumed image shape: {assert_img_shape}")
    chk_results = check_multiple_img_dim(in_files, assert_img_shape)

    imgs_array = []
    for i, f in enumerate(in_files):
        img = load_img(f.as_posix())
        img_array = img.get_fdata()
        img_array = np.rot90(img_array, 1)
        imgs_array.append(img_array)
    return np.array(imgs_array)


def read_imgs(in_files):
    imgs_array = []
    for i, f in enumerate(in_files):
        img = load_img(f.as_posix())
        img_array = img.get_fdata()
        img_array = np.rot90(img_array, 1)
        imgs_array.append(img_array)
    return np.array(imgs_array)


def read_imgs_hc_alff():
    # target_path = DATAPATH[0]+MODELITIES[1]+"/"+GROUPS[0]+"/"+MEASURES[0]
    # quary_pattern = "**/mfALFFMap*"
    # assert_n_subjects = 44 # hc
    # assert_img_shape = (53, 63, 46)
    # _imgs_array = read_imgs_old(target_path, quary_pattern, assert_n_subjects, assert_img_shape)
    in_files, _, _, _ = p_datasets.retrive_raw_data_bold_ctrl_alff(logger)
    imgs_array = read_imgs(in_files)
    return np.array(imgs_array)


def read_imgs_02d_alff():
    # target_path = DATAPATH[0]+MODELITIES[1]+"/"+GROUPS[1]+"/"+MEASURES[2]
    # quary_pattern = "**/mfALFFMap*"
    # assert_n_subjects = 62 # 02D
    # assert_img_shape = (53, 63, 46)
    # imgs_array = read_imgs_old(target_path, quary_pattern, assert_n_subjects, assert_img_shape)
    in_files, _, _, _ = p_datasets.retrive_raw_data_bold_02d_alff(logger)
    imgs_array = read_imgs(in_files)
    return np.array(imgs_array)


def read_imgs_03id_alff():
    # target_path = DATAPATH[0]+MODELITIES[1]+"/"+GROUPS[2]+"/"+MEASURES[4]
    # quary_pattern = "**/mfALFFMap*"
    # assert_n_subjects = 48 # 03ID
    # assert_img_shape = (53, 63, 46)
    # imgs_array = read_imgs_old(target_path, quary_pattern, assert_n_subjects, assert_img_shape)
    in_files, _, _, _ = p_datasets.retrive_raw_data_bold_03id_alff(logger)
    imgs_array = read_imgs(in_files)
    return np.array(imgs_array)


def read_imgs_04sh_alff():
    # target_path = DATAPATH[0]+MODELITIES[1]+"/"+GROUPS[3]+"/"+MEASURES[6]
    # quary_pattern = "**/mfALFFMap*"
    # assert_n_subjects = 33 # 04_SH
    # assert_img_shape = (53, 63, 46)
    # imgs_array = read_imgs_old(target_path, quary_pattern, assert_n_subjects, assert_img_shape)
    in_files, _, _, _ = p_datasets.retrive_raw_data_bold_04sh_alff(logger)
    imgs_array = read_imgs(in_files)
    return np.array(imgs_array)


def prepare_data_concatenated():
    hc_imgs = read_imgs_hc_alff()
    hc_labels = np.zeros((hc_imgs.shape[0], hc_imgs.shape[3]))
    dep_imgs = read_imgs_02d_alff()
    dep_labels = np.ones((dep_imgs.shape[0], dep_imgs.shape[3]))
    id_imgs = read_imgs_03id_alff()
    id_labels = np.ones((id_imgs.shape[0], id_imgs.shape[3]))
    sh_imgs = read_imgs_04sh_alff()
    sh_labels = np.ones((sh_imgs.shape[0], sh_imgs.shape[3]))
    all_imgs = np.concatenate((hc_imgs, dep_imgs, id_imgs, sh_imgs), axis=0)
    all_labels = np.concatenate((hc_labels, dep_labels, id_labels, sh_labels), axis=0)
    return all_imgs, all_labels


def prepare_data_concatenated_3view(view_key, padded_shape=None):
    view_dict = {"cronal": 1, "sagittal": 2, "axial": 3}
    view_id = view_dict[view_key]
    hc_imgs = read_imgs_hc_alff()
    hc_labels = np.zeros((hc_imgs.shape[0], hc_imgs.shape[view_id]))
    dep_imgs = read_imgs_02d_alff()
    dep_labels = np.ones((dep_imgs.shape[0], dep_imgs.shape[view_id]))
    id_imgs = read_imgs_03id_alff()
    id_labels = np.ones((id_imgs.shape[0], id_imgs.shape[view_id]))
    sh_imgs = read_imgs_04sh_alff()
    sh_labels = np.ones((sh_imgs.shape[0], sh_imgs.shape[view_id]))
    all_imgs = np.concatenate((hc_imgs, dep_imgs, id_imgs, sh_imgs), axis=0)
    all_labels = np.concatenate((hc_labels, dep_labels, id_labels, sh_labels), axis=0)
    return all_imgs, all_labels


def make_padding(img_to_pad, padded_shape, view_key):
    def _get_pad_width(length):
        width = padded_shape - length
        modulo = width%2
        quotient = width//2
        if modulo == 0:
            return (quotient, quotient)
        else:
            return (quotient, quotient + 1)

    view_field = {"cronal": (46, 53), "sagittal": (46, 63), "axial": (63, 53)}
    pad_width = ((0, 0),
                 _get_pad_width(view_field[view_key][0]),
                 _get_pad_width(view_field[view_key][1]),
                 (0, 0))
    logger.info("Start padding...")
    padded_img = np.pad(img_to_pad, pad_width,
                        mode="constant", constant_values=(0, 0))
    return padded_img


def prepare_split_data_axial(all_imgs, all_labels, test_size=0.05, padded_shape=None):
    logger.info("Axial view")
    X_train, X_test, y_train, y_test = train_test_split(all_imgs, all_labels, test_size=test_size)
    X_train = np.swapaxes(X_train, 1, 3)
    X_train = np.swapaxes(X_train, 2, 3)
    X_test = np.swapaxes(X_test, 1, 3)
    X_test = np.swapaxes(X_test, 2, 3)
    X_train = X_train.reshape(-1, 63, 53, 1)
    X_test = X_test.reshape(-1, 63, 53, 1)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    logger.info(f"Number of image - train: {X_train.shape[0]}; Img shape: {X_train.shape[1:4]}")
    logger.info(f"Number of image - test: {X_test.shape[0]}; Img shape: {X_test.shape[1:4]}")
    y_train = y_train.reshape((-1))
    y_test = y_test.reshape((-1))
    logger.info(f"Number of labels - train: {y_train.shape[0]}")
    logger.info(f"Number of labels - test: {y_test.shape[0]}")
    if padded_shape:
        X_train = make_padding(X_train, padded_shape, view_key="axial")
        X_test = make_padding(X_test, padded_shape, view_key="axial")
        logger.info("After padding...")
        logger.info(f"Number of image - train: {X_train.shape[0]}; Img shape: {X_train.shape[1:4]}")
        logger.info(f"Number of image - test: {X_test.shape[0]}; Img shape: {X_test.shape[1:4]}")
    return X_train, X_test, y_train, y_test


def prepare_split_data_cronal(all_imgs, all_labels, test_size=0.05, padded_shape=None):
    logger.info("Cronal view")
    X_train, X_test, y_train, y_test = train_test_split(all_imgs, all_labels, test_size=test_size)
    X_train = np.swapaxes(X_train, 2, 3)
    X_train = np.flip(X_train, axis=(2,3))
    X_test = np.swapaxes(X_test, 2, 3)
    X_test = np.flip(X_test, axis=(2,3))
    X_train = X_train.reshape(-1, 46, 53, 1)
    X_test = X_test.reshape(-1, 46, 53, 1)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    logger.info(f"Number of image - train: {X_train.shape[0]}; Img shape: {X_train.shape[1:4]}")
    logger.info(f"Number of image - test: {X_test.shape[0]}; Img shape: {X_test.shape[1:4]}")
    y_train = y_train.reshape((-1))
    y_test = y_test.reshape((-1))
    logger.info(f"Number of labels - train: {y_train.shape[0]}")
    logger.info(f"Number of labels - test: {y_test.shape[0]}")
    if padded_shape:
        X_train = make_padding(X_train, padded_shape, view_key="cronal")
        X_test = make_padding(X_test, padded_shape, view_key="cronal")
        logger.info("After padding...")
        logger.info(f"Number of image - train: {X_train.shape[0]}; Img shape: {X_train.shape[1:4]}")
        logger.info(f"Number of image - test: {X_test.shape[-1]}; Img shape: {X_test.shape[1:4]}")
    return X_train, X_test, y_train, y_test


def prepare_split_data_sagittal(all_imgs, all_labels, test_size=0.05, padded_shape=None):
    logger.info("Sagittal view")
    X_train, X_test, y_train, y_test = train_test_split(all_imgs, all_labels, test_size=test_size)
    X_train = np.swapaxes(X_train, 1, 2)
    X_train = np.swapaxes(X_train, 2, 3)
    X_train = np.flip(X_train, axis=(2,3))
    X_test = np.swapaxes(X_test, 1, 2)
    X_test = np.swapaxes(X_test, 2, 3)
    X_test = np.flip(X_test, axis=(2,3))
    X_train = X_train.reshape(-1, 46, 63, 1)
    X_test = X_test.reshape(-1, 46, 63, 1)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    logger.info(f"Number of image - train: {X_train.shape[0]}; Img shape: {X_train.shape[1:4]}")
    logger.info(f"Number of image - test: {X_test.shape[0]}; Img shape: {X_test.shape[1:4]}")
    y_train = y_train.reshape((-1))
    y_test = y_test.reshape((-1))
    logger.info(f"Number of labels - train: {y_train.shape[0]}")
    logger.info(f"Number of labels - test: {y_test.shape[0]}")
    if padded_shape:
        X_train = make_padding(X_train, padded_shape, view_key="sagittal")
        X_test = make_padding(X_test, padded_shape, view_key="sagittal")
        logger.info("After padding...")
        logger.info(f"Number of image - train: {X_train.shape[0]}; Img shape: {X_train.shape[1:4]}")
        logger.info(f"Number of image - test: {X_test.shape[0]}; Img shape: {X_test.shape[1:4]}")
    # data = X_test
    # examine_img(data[19,:,:,0])
    # __import__('ipdb').set_trace()
    return X_train, X_test, y_train, y_test


def build_model_basic(X_train, X_test, y_train, y_test, view_key=None):
    view_field = {"cronal": (46, 53), "sagittal": (46, 63), "axial": (63, 53)}
    input_shape = (63, 53, 1)
    if view_key:
        input_shape = (view_field[view_key][0], view_field[view_key][1], 1)
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # Take a look at the model summary
    model.summary()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train,
            y_train,
            batch_size=64,
            epochs=10,
            validation_data=(X_test, y_test))


def build_model_basic_autokeras(X_train, X_test, y_train, y_test, view_key=None):
    view_field = {"cronal": (46, 53), "sagittal": (46, 63), "axial": (63, 53)}
    input_shape = (63, 53, 1)
    if view_key:
        input_shape = (view_field[view_key][0], view_field[view_key][1], 1)

    import autokeras as ak
    clf = ak.ImageClassifier(verbose=True)
    clf.fit(X_train, y_train)
    clf.final_fit(X_train, y_train, X_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    #results = clf.predict(X_test)
    results = clf.evaluate(x_test, y_test)
    print(results)

    #clf.fit(x_train, y_train, time_limit=11 * 60 * 60)
    #clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    #print(y)


def build_model_3view(X_train, X_test, y_train, y_test, input_shape):
    """Note:
        1s 166us/sample - loss: 0.5579 - acc: 0.7571 - val_loss: 0.3926 - val_acc: 0.9000"""
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # Take a look at the model summary
    model.summary()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train,
            y_train,
            batch_size=64,
            epochs=10,
            validation_data=(X_test, y_test))


def build_model_3view_autokeras(X_train, X_test, y_train, y_test, input_shape):
    import autokeras as ak
    clf = ak.ImageClassifier(verbose=True)
    clf.fit(X_train, y_train)
    clf.final_fit(X_train, y_train, X_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    #results = clf.predict(X_test)
    results = clf.evaluate(x_test, y_test)
    print(results)


def prepare_data_concatenated_padding_3view():
    view_key_list = ["axial", "cronal", "sagittal"]
    fun_dict = {"axial": prepare_split_data_axial, "cronal": prepare_split_data_cronal, "sagittal": prepare_split_data_sagittal}
    padded_shape = 63
    test_size = 0.1
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for i, view_key in enumerate(view_key_list):
        all_imgs, all_labels = prepare_data_concatenated_3view(view_key)
        X_train, X_test, y_train, y_test = fun_dict[view_key](
            all_imgs, all_labels, test_size=test_size, padded_shape=padded_shape)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    logger.info("Concatenate data from 3 views")
    X_train = np.concatenate(X_trains, axis=0)
    X_test = np.concatenate(X_tests, axis=0)
    logger.info(f"Number of image - train: {X_train.shape[0]}; Img shape: {X_train.shape[1:4]}")
    logger.info(f"Number of image - test: {X_test.shape[0]}; Img shape: {X_test.shape[1:4]}")
    y_train = np.concatenate(y_trains)
    y_test = np.concatenate(y_tests)
    logger.info(f"Number of labels - train: {y_train.shape[0]}")
    logger.info(f"Number of labels - test: {y_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def main_basic_cnn_cla_orig():
    """Note:
        1s 166us/sample - loss: 0.5579 - acc: 0.7571 - val_loss: 0.3926 - val_acc: 0.9000"""
    view_key = "axial"
    all_imgs, all_labels = prepare_data_concatenated_3view(view_key)
    X_train, X_test, y_train, y_test = prepare_split_data_axial(all_imgs, all_labels)
    #examine_img(data[1,:,:,0])
    build_model_basic(X_train, X_test, y_train, y_test)


def main_basic_cnn_cla():
    #view_key = "axial"
    """1s 166us/sample - loss: 0.5579 - acc: 0.7571 - val_loss: 0.3926 - val_acc: 0.9000"""
    #view_key = "cronal"
    """2s 159us/sample - loss: 0.5495 - acc: 0.7627 - val_loss: 0.5034 - val_acc: 0.8000"""
    view_key = "sagittal"
    """2s 166us/sample - loss: 0.5567 - acc: 0.7571 - val_loss: 0.3909 - val_acc: 0.9000"""
    fun_dict = {"axial": prepare_split_data_axial, "cronal": prepare_split_data_cronal, "sagittal": prepare_split_data_sagittal}
    padded_shape = None
    test_size = 0.05
    all_imgs, all_labels = prepare_data_concatenated_3view(view_key)
    X_train, X_test, y_train, y_test = fun_dict[view_key](
        all_imgs, all_labels, test_size=test_size, padded_shape=padded_shape)
    build_model_basic(X_train, X_test, y_train, y_test, view_key)


def main_basic_cnn_cla_auto():
    view_key = "axial"
    """2s 166us/sample - loss: 0.5567 - acc: 0.7571 - val_loss: 0.3909 - val_acc: 0.9000"""
    fun_dict = {"axial": prepare_split_data_axial, "cronal": prepare_split_data_cronal, "sagittal": prepare_split_data_sagittal}
    padded_shape = None
    test_size = 0.05
    all_imgs, all_labels = prepare_data_concatenated_3view(view_key)
    X_train, X_test, y_train, y_test = fun_dict[view_key](
        all_imgs, all_labels, test_size=test_size, padded_shape=padded_shape)
    #build_model_basic(X_train, X_test, y_train, y_test, view_key)
    build_model_basic_autokeras(X_train, X_test, y_train, y_test, view_key)


def main_basic_cnn_3view_cla():
    """Note:
        6s 218us/sample - loss: 0.5541 - acc: 0.7583 - val_loss: 0.4225 - val_acc: 0.8778"""
    input_shape=(63, 63, 1)
    X_train, X_test, y_train, y_test = prepare_data_concatenated_padding_3view()
    build_model_3view(X_train, X_test, y_train, y_test, input_shape)


def main_basic_cnn_3view_cla_autokeras():
    input_shape=(63, 63, 1)
    X_train, X_test, y_train, y_test = prepare_data_concatenated_padding_3view()
    build_model_3view_autokeras(X_train, X_test, y_train, y_test, input_shape)


if __name__ == "__main__":
    main_basic_cnn_cla()
    #main_basic_cnn_cla_auto()
    #main_basic_cnn_3view_cla_autokeras()
    #main_basic_cnn_3view_cla()
