from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

def main():
    # Off-the-shelf ResNet-50.
    # TODO Replace with ResNext w/ multiple tasks (gender, age, etc.)
    model = resnet50.ResNet50(include_top=True, weights=None, classes=2)
    model.compile(optimizer='adam', # TODO: replace with tuned SGD
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    img_gen = ImageDataGenerator(samplewise_center=True)
    train_gen = img_gen.flow_from_directory(directory="data/processed/train",
                                            target_size=(224, 224),
                                            classes=['f', 'm'],
                                            class_mode="categorical",
                                            batch_size=32,
                                            seed=0)

    test_gen = img_gen.flow_from_directory(directory="data/processed/test",
                                           target_size=(224, 224),
                                           classes=['f', 'm'],
                                           class_mode="categorical",
                                           batch_size=32,
                                           seed=0)

    checkpointer = ModelCheckpoint(filepath='best-weights.h5', verbose=1,
                                   save_best_only=True)

    model.fit_generator(train_gen, 343,
                        validation_data=test_gen,
                        validation_steps=80,
                        epochs=1500,
                        use_multiprocessing=True,
                        callbacks=[checkpointer])

if __name__ == '__main__':
    main()