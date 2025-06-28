import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    'dataset/test', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# Load best model
model = tf.keras.models.load_model('butterfly_model_best.h5')

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')

# Classification report
class_names = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Optional: Fine-tuning
def fine_tune_model():
    model = tf.keras.models.load_model('butterfly_model_best.h5')
    # Unfreeze top 20 layers
    for layer in model.layers[-20:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    model.save('butterfly_model_finetuned.h5')
    print("Fine-tuning completed.")

# Uncomment to fine-tune if accuracy is low
# fine_tune_model()

print("Model evaluation completed. Results saved.")