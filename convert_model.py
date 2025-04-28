import tensorflow as tf

# Charger le modèle original
model_path = 'models/vgg16_finetuned.keras'
try:
    model = tf.keras.models.load_model(model_path)
    # Sauvegarder en format SavedModel
    model.save('models/vgg16_saved_model', save_format='tf')
    print("Modèle converti avec succès!")
except Exception as e:
    print(f"Erreur lors de la conversion: {e}")