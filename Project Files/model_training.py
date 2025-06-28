import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data generators from Step 2
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
).flow_from_directory('dataset/train', target_size=(224, 224), batch_size=32, class_mode='categorical')

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    'dataset/val', target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Load model from Step 3
model = tf.keras.models.load_model('butterfly_model_initial.h5')

# Recompile the model to ensure eager execution is enabled
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('butterfly_model_best.h5', monitor='val_accuracy', save_best_only=True)

# Train model
history = model.fit(
    train_generator,
    epochs=5, # Restored to 5 epochs
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)

# Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Model training completed. Best model saved as 'butterfly_model_best.h5'.")