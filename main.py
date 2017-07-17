from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau


def main():
    # Off-the-shelf ResNet-50.
    # TODO Replace with ResNext w/ multiple tasks (gender, age, etc.)
    model = resnet50.ResNet50(include_top=True, weights=None, classes=2)

    batch_size = 64
    opt = SGD(lr=0.1, momentum=0.9, nesterov=True)
    lr_reduction = ReduceLROnPlateau(factor=0.1, verbose=1)
    model.compile(optimizer=opt,  # TODO: tune SGD parameters for better perf.
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # TODO: data augmentation
    img_gen = ImageDataGenerator(samplewise_center=True)
    train_gen = img_gen.flow_from_directory(directory="data/processed/train",
                                            target_size=(224, 224),
                                            classes=['f', 'm'],
                                            class_mode="categorical",
                                            batch_size=batch_size,
                                            seed=0)

    test_gen = img_gen.flow_from_directory(directory="data/processed/test",
                                           target_size=(224, 224),
                                           classes=['f', 'm'],
                                           class_mode="categorical",
                                           batch_size=batch_size,
                                           seed=0)

    checkpointer = ModelCheckpoint(filepath='best-weights.h5', verbose=1,
                                   save_best_only=True)

    # TODO: multi-gpu training
    model.fit_generator(train_gen, 10945 // batch_size + 1,
                        validation_data=test_gen,
                        validation_steps=2558 // batch_size + 1,
                        epochs=500,
                        use_multiprocessing=True,
                        callbacks=[checkpointer, lr_reduction])

if __name__ == '__main__':
    main()