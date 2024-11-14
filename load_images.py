
import tensorflow as tf

def load_image_data(directory, img_size=(224, 224), batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size
    )
    return dataset


def resize_images(dataset, target_size=(224, 224)):
    resized_images = dataset.map(lambda x, y: (tf.image.resize(x, target_size), y))
    return resized_images




if __name__ == "__main__":
    dataset = load_image_data(r"C:\Users\Pushkraj\OneDrive\Desktop\Tuberculosis\images")

    resized_dataset = resize_images(dataset)


    for images, labels in dataset.take(1):
        print(images.shape)
        print(labels.shape)



