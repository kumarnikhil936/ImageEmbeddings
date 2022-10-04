import numpy as np
import time
from pathlib import Path
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import features as features_lib
from tensorflow_datasets.core import dataset_utils

from efficientnet.preprocessing import center_crop_and_resize
from efficientnet.tfkeras import EfficientNetB0

import pyarrow.parquet as pq
import pyarrow as pa
import faiss

import IPython
from IPython.display import display
from ipywidgets import widgets, HBox, VBox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def fetch_save_images(dataset='tf_flowers', 
                      output_folder='./images', 
                      images_count=1000, 
                      num_examples=300):
    # fetch dataset
    ds, ds_info = tfds.load('tf_flowers', split="train", with_info=True)
    print(ds_info)

    # create the output folder 
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # get image and label key
    image_keys = [k for k,feature in ds_info.features.items() if isinstance(feature, features_lib.Image)]
    image_key = image_keys[0] if len(image_keys) == 1 else None

    label_keys = [k for k, feature in ds_info.features.items() if isinstance(feature, features_lib.ClassLabel)]
    label_key = label_keys[0] if len(label_keys) == 1 else None

    # save some sample images 
    examples = list(dataset_utils.as_numpy(ds.take(num_examples)))

    for i, ex in enumerate(examples):
        # get image data from the dict
        image = ex[image_key]
        if len(image.shape) != 3:
            raise ValueError("Image dimension should be 3. tfds.show_examples does not support batched examples.")

        _, _, c = image.shape    
        if c == 1:
            image = image.reshape(image.shape[:2])

        image = center_crop_and_resize(image, 224).astype(np.uint8)
        im = Image.fromarray(image)

        if label_key:
            label = ex[label_key]
            label_str = ds_info.features[label_key].int2str(label).replace("/", "_")
        else:
            label_str = ""

        # save the image    
        im.save(f"{output_folder}/image_{label_str}_{i}.jpeg")
        
    return ds_info, ds


def list_files(images_path):
    return tf.data.Dataset.list_files(images_path + '/*', shuffle=False).cache()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, image_name):
    feature = {
        'image_name': _bytes_feature(image_name),
        'image': _bytes_feature(image)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(image, image_name):
    tf_string = tf.py_function(serialize_example, (image, image_name), tf.string)
    return tf.reshape(tf_string, ())


def process_path(file_path):
    parts = tf.strings.split(file_path, '/')
    image_name = tf.strings.split(parts[-1], '.')[0]
    raw = tf.io.read_file(file_path)
    return raw, image_name


def read_image_file_write_tfrecord(files_ds, output_filename):
    image_ds = files_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    serialized_features_dataset = image_ds.map(tf_serialize_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    writer = tf.data.experimental.TFRecordWriter(output_filename)
    # writer = tf.io.TFRecordWriter(output_filename)
    writer.write(serialized_features_dataset)

    
def image_files_to_tfrecords(list_ds, output_folder, num_shard):
    start = time.time()
    for shard_id in range(0, num_shard):
        shard_list = list_ds.shard(num_shards=num_shard, index=shard_id)
        output_filename = output_folder + "/part-" + "{:03d}".format(shard_id) + ".tfrecord"
        read_image_file_write_tfrecord(shard_list, output_filename)
        print("Shard " + str(shard_id) + " saved after " + str(int(time.time() - start)) + "s")
        

def write_tfrecord(image_folder, output_folder, num_shard=100):
    list_ds = list_files(image_folder)
    image_files_to_tfrecords(list_ds, output_folder, num_shard)
    
    
def _parse_function(example_proto):
    feature_description = {
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def preprocess_image(d):
    image_name = d['image_name']
    raw = d['image']
    image = tf.image.decode_jpeg(raw)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, image_name


def read_tfrecord(filename):
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    return raw_dataset \
        .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .apply(tf.data.experimental.ignore_errors())


def images_to_embeddings(model, dataset, batch_size):
    return model.predict(dataset.batch(batch_size).map(lambda image_raw, image_name: image_raw), verbose=1)


def save_embeddings_ds_to_parquet(embeddings, dataset, path):
    embeddings = pa.array(embeddings.tolist(), type=pa.list_(pa.float32()))
    image_names = pa.array(dataset.map(lambda image_raw, image_name: image_name).as_numpy_iterator())
    table = pa.Table.from_arrays([image_names, embeddings], ["image_name", "embedding"])
    pq.write_table(table, path)
    

def tfrecords_to_write_embeddings(tfrecords_folder, output_folder, model, batch_size):
    tfrecords = [f.numpy().decode("utf-8") for f in tf.data.Dataset.list_files(tfrecords_folder + "/*.tfrecord", shuffle=False)]
    
    start = time.time()
    for shard_id, tfrecord in enumerate(tfrecords):
        shard = read_tfrecord(tfrecord)
        embeddings = images_to_embeddings(model, shard, batch_size)
        print("Shard " + str(shard_id) + " done after " + str(int(time.time() - start)) + "s")
        
        save_embeddings_ds_to_parquet(embeddings, shard, output_folder + "/part-" + "{:03d}".format(shard_id) + ".parquet")
        print("Shard " + str(shard_id) + " saved after " + str(int(time.time() - start)) + "s")


# run inference, generate embeddings and write to parquets
def run_inference(tfrecords_folder, output_folder, batch_size=1000):
    model = EfficientNetB0(weights='imagenet', include_top=False, pooling="avg")
    tfrecords_to_write_embeddings(tfrecords_folder, output_folder, model, batch_size)
    
    
# knn search
def search(emb, k=5):
    D, I = index.search(np.expand_dims(emb, 0), k)
    return list(zip(D[0], [id_to_name[x] for x in I[0]]))

# show original image
def display_picture(image_name):
    display(IPython.display.Image(filename=f"{image_name}.jpeg"))

# show similar images
def display_results(results):
    hbox = HBox([VBox([widgets.Label(f"{distance:.2f} {image_name}"), widgets.Image(value=open(f"{image_name}.jpeg", 'rb').read())]) for distance, image_name in results])
    display(hbox)
    
    
if __name__ == '__main__':
    ds_info, ds = fetch_save_images()
    fig = tfds.show_examples(ds, ds_info)

    image_folder = "images"
    output_folder = "tfrecords"

    write_tfrecord(image_folder, output_folder, 1)

    run_inference("tfrecords", "embeddings", 1000)

    embeddings = pq.read_table("embeddings").to_pandas()
    print(embeddings.head())

    print('Image name:', str(embeddings['image_name'].iloc[0]), 
          '\nEmbedding:', embeddings['embedding'].iloc[0], 
          '\nLength of embedding:', len(embeddings['embedding'].iloc[0]))

    id_to_name = {k:v.decode("utf-8") for k,v in enumerate(list(embeddings["image_name"]))}
    name_to_id = {v:k for k,v in id_to_name.items()}

    embeddings_stacked = np.stack(embeddings["embedding"].to_numpy())
    xb = embeddings_stacked

    length_embedding = 1280
    index = faiss.IndexFlatIP(length_embedding)
    index.add(xb)

    image_id = np.random.randint(0, max(id_to_name.keys()))
    print(image_id, id_to_name[image_id])
    
    display_picture(id_to_name[image_id])
    display_results(search(xb[image_id]))