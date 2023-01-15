import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory as imdd
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping


train_ds = imdd(
    r'/home/dell/ML/Hand/asl_alphabet_train (1)/asl_alphabet_train-20230111T114307Z-001/asl_alphabet_train (1)/',
    color_mode = 'grayscale',
    label_mode = 'categorical',
    shuffle = True,
    image_size = (128,128),
    seed = 123,
    # class_names = class_name,
    subset = 'training',
    validation_split = 0.2,
)
val_ds = imdd(
    r'/home/dell/ML/Hand/asl_alphabet_train (1)/asl_alphabet_train-20230111T114307Z-001/asl_alphabet_train (1)/',
    color_mode = 'grayscale',
    label_mode = 'categorical',
    shuffle = True,
    image_size = (128,128),
    seed = 123,
    # class_names = class_name,
    subset = 'validation',
    validation_split = 0.2,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential(
    [
        layers.Rescaling(1./255., input_shape = (128,128,1)),
        layers.Conv2D(4, kernel_size = (5,5), padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(8, kernel_size = (3,3), padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, kernel_size = (3,3), padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(576, activation = 'relu'),
        layers.Dropout(0.05),
        layers.Dense(288, activation = 'relu'),
        # layers.Dropout(0.3),
        layers.Dense(144, activation = 'relu'),
        layers.Dropout(0.3),
        layers.Dense(72, activation = 'relu'),
        # layers.Dropout(0.1),
        layers.Dense(36 , activation = 'relu'),
        layers.Dropout(0.05),
        layers.Dense(29, name = 'output', activation = 'softmax')
    ]
)

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = ['accuracy']
)

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
model.fit(
    train_ds,
    validation_data = val_ds,
    callbacks = [early_stop],
    epochs = 32
)

model.save(
    '/home/dell/ML/Hand/model.h5'
)