import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from numpy import ndarray

IMG_SIZE = (32, 32)
# DATASET_PATH = "data/mini_speech_commands"
DATASET_PATH = "data/speech_commands"
data_directory = pathlib.Path(DATASET_PATH)
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 15


def read_commands(data_dir) -> ndarray:
    commands_ = np.array(tf.io.gfile.listdir(str(data_dir)))
    print(commands_)
    return commands_


def get_shuffled_filenames(data_dir) -> list[str]:
    filenames_ = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames_ = tf.random.shuffle(filenames_)
    num_samples = len(filenames_)
    print("Size: ", num_samples)
    print("Random file: ", filenames_[0])
    return filenames_


def split_dataset(filenames_):
    fs_length = len(filenames_)
    percent80 = int(fs_length * 0.8)
    percent10 = int(fs_length * 0.1)
    train_files = filenames_[:percent80]
    validation_files = filenames_[percent80: percent80 + percent10]
    test_files = filenames_[-percent10:]
    print(len(train_files), train_files[0])
    print(len(validation_files), validation_files[0])
    print(len(test_files), test_files[0])
    return train_files, validation_files, test_files


def decode_audio(audio_file):
    audio_, _ = tf.audio.decode_wav(contents=audio_file)
    return tf.squeeze(audio_, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    label_ = get_label(file_path)
    audio_bin = tf.io.read_file(file_path)
    waveform_ = decode_audio(audio_bin)
    # print(label_)
    # print(waveform_.shape)
    return waveform_, label_


def create_waveform_ds(filenames_):
    files_ds = tf.data.Dataset.from_tensor_slices(filenames_)
    waveform_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE
    )
    return waveform_ds


def print_wav_ds(wav_ds_):
    fig, axes = plt.subplots(3, 3, figsize=(10, 12))
    for i, (audio, label) in enumerate(wav_ds_.take(9)):
        ax = axes[i // 3][i % 3]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)
    plt.show()


def get_spectrogram(waveform_):
    input_len = 16000
    waveform_ = waveform_[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform_), dtype=tf.float32)
    waveform_ = tf.cast(waveform_, dtype=tf.float32)
    equal_length = tf.concat([waveform_, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def play_audio(waveform_ds):
    for wav, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(wav)
        print('Label: ', label)
        print('Shape: ', wav.shape)
        print('Spectrogram: ', spectrogram.shape)
        display.display(display.Audio(wav, rate=16000))


def plot_spectrogram_axes(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def plot_spectrogram(waveform_):
    for wav, label in waveform_.take(1):
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(wav.shape[0])
        axes[0].plot(timescale, wav.numpy())
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])
        plot_spectrogram_axes(get_spectrogram(wav), axes[1])
        axes[1].set_title('Spectrogram ' + str(label.numpy().decode()))
        plt.show()


def plot_spectrogram_full(spectrogram):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    plt.pcolormesh(X, Y, log_spec)
    plt.show()


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def get_spectrogram_ds(waveform_ds):
    spectrogram_ds = waveform_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE
    )
    return spectrogram_ds


def print_spectrogram_ds(spectrogram_ds_, commands):
    fig, axes = plt.subplots(3, 3, figsize=(10, 12))
    for i, (spec, l_id) in enumerate(spectrogram_ds_.take(9)):
        ax = axes[i // 3][i % 3]
        plot_spectrogram_axes(spec, ax)
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        ax.set_title(str(l_id.numpy()) + ' ' + commands[l_id.numpy()])
    plt.show()


def prepare_to_batch(spec_ds):
    batch_size = 64
    spec_ds = spec_ds.batch(batch_size)
    spec_ds = spec_ds.cache().prefetch(AUTOTUNE)
    return spec_ds


def prepare_to_spec(tr_audio_ds, va_audio_ds, te_audio_ds):
    tr_wav_ds = create_waveform_ds(tr_audio_ds)
    tr_spectrogram_ds = get_spectrogram_ds(tr_wav_ds)
    tr_spectrogram_ds = prepare_to_batch(tr_spectrogram_ds)

    va_wav_ds = create_waveform_ds(va_audio_ds)
    va_spectrogram_ds = get_spectrogram_ds(va_wav_ds)
    va_spectrogram_ds = prepare_to_batch(va_spectrogram_ds)

    te_wav_ds = create_waveform_ds(te_audio_ds)
    te_spectrogram_ds = get_spectrogram_ds(te_wav_ds)
    te_spectrogram_ds = prepare_to_batch(te_spectrogram_ds)

    return tr_spectrogram_ds, va_spectrogram_ds, te_spectrogram_ds


def return_spec(spec, label):
    return spec


def create_model_for_audio(spectrogram_ds_, commands):
    global input_shape
    for spectrogram, _ in spectrogram_ds_.take(1):
        input_shape = spectrogram.shape[1:]
    print("INPUT SHAPE: ", input_shape)
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(data=spectrogram_ds_.map(map_func=return_spec))
    model_ = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Resizing(32, 32),
        norm_layer,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(commands)),
    ])
    model_.summary()
    model_.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model_


def train_for_audio(model_, t_ds_, v_ds_, epochs_):
    history = model_.fit(
        t_ds_,
        validation_data=v_ds_,
        epochs=epochs_,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
    )
    metrics = history.history
    # plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    # plt.legend(['loss', 'val_loss'])
    # plt.show()


def test_for_audio(model_, commands, test_ds):
    sum_ = 0.0
    count = 0
    for spectrogram, label in test_ds.take(len(test_ds)):
        predicted_batch = model_.predict_on_batch(spectrogram)
        for i in range(len(predicted_batch)):
            y_pred = commands[np.argmax(predicted_batch[i])]
            y_true = commands[label[i]]
            sum_ += y_pred == y_true
            count += 1
    test_acc = sum_ / count
    print(f'Test set accuracy: {test_acc:.0%}')


################################################################################################
# def load_from_dir(data_dir, subset_name):
#     return tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset=subset_name, seed=1337,
#                                                        image_size=IMG_SIZE)


# def prepare_data(path: str) -> tuple:
#     data_dir = pathlib.Path(path)
#     return load_from_dir(data_dir, "training"), load_from_dir(data_dir, "validation")


# def read_image(path_to_image: str):
#     img = tf.keras.preprocessing.image.load_img(path_to_image, target_size=IMG_SIZE)
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     return np.vstack([img])


# def prepate_data_for_test(path: str) -> tuple:
#     data_dir = pathlib.Path(path)
#     test_images = []
#     test_labels = []
#     for dir_ in os.listdir(data_dir):
#         for image in glob.glob(path + "/" + str(dir_) + "/*.png"):
#             test_images.append(read_image(str(image)))
#             test_labels.append(str(dir_))
#     return test_images, test_labels


# def configure(ds_list: list) -> None:
#     AUTOTUNE = tf.data.AUTOTUNE
#     for ds in ds_list:
#         ds = ds.cache().prefetch(buffer_size=AUTOTUNE).shuffle(buffer_size=570)


# def create_model():
#     model_ = tf.keras.Sequential([
#         tf.keras.layers.Rescaling(1. / 255),
#         tf.keras.layers.Conv2D(32, 3, activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Conv2D(64, 3, activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Conv2D(128, 3, activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(11)
#     ])
#     model_.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy'])
#     return model_


# def train(model_, t_ds_, v_ds_, epochs_):
#     history = model_.fit(
#         t_ds_,
#         validation_data=v_ds_,
#         epochs=epochs_
#     )
#     metrics = history.history
#     # plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
#     # plt.legend(['loss', 'val_loss'])
#     # plt.show()


# def test(model_, path: str, t_ds_) -> None:
#     test_images, test_labels = prepate_data_for_test(path)
#     sum_ = 0.0
#     for i in range(len(test_images)):
#         predicted = model_.predict_on_batch(test_images[i])
#         y_pred = t_ds_.class_names[np.argmax(predicted)]
#         y_true = test_labels[i]
#         sum_ += y_pred == y_true
#     test_acc = sum_ / len(test_labels)
#     print(f'Test set accuracy: {test_acc:.0%}')


if __name__ == '__main__':
    filenames = get_shuffled_filenames(data_directory)
    commands = read_commands(data_directory)
    tr_ds, va_ds, te_ds = split_dataset(filenames)
    tr_spec_ds, va_spec_ds, te_spec_ds = prepare_to_spec(tr_ds, va_ds, te_ds)
    model = create_model_for_audio(tr_spec_ds, commands)
    train_for_audio(model, tr_spec_ds, va_spec_ds, EPOCHS)
    test_for_audio(model, commands, te_spec_ds)

    # print_wav_ds(tr_wav_ds)
    # play_audio(tr_wav_ds)
    # plot_spectrogram(tr_wav_ds)
    # print_spectrogram_ds(tr_spec_ds, commands)

    # t_ds, v_ds = prepare_data("dataset_smiles/DataSet")
    # configure([t_ds, v_ds])
    # model = create_model()
    # train(model, t_ds, v_ds, 10)
    # model.evaluate(t_ds, verbose=2)
    # test(model, "dataset_smiles/TestSet", t_ds)
