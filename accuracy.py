from tensorflow.keras.models import load_model

# Load the model in the original environment
model = load_model('model_update_legacy.h5')

# Re-save it in a compatible format
model.save('model_compatible.h5')
