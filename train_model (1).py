"""
SpectraMining - CNN Model Training Script
Train a deep learning model to classify mining vs natural land using transfer learning
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import ee
from datetime import datetime, timedelta
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model parameters
IMG_SIZE = 224  # ResNet50 input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Paths
MODEL_SAVE_PATH = "spectramining_model.h5"
DATASET_PATH = "training_data/"  # You'll need to create this

# ============================================================================
# GOOGLE EARTH ENGINE DATASET GENERATOR
# ============================================================================

def initialize_gee():
    """Initialize Google Earth Engine for data collection"""
    try:
        ee.Initialize()
        print("✅ Google Earth Engine initialized successfully")
        return True
    except:
        print("❌ Please authenticate GEE first: earthengine authenticate")
        return False

def fetch_training_patch(lat, lon, label, size=224):
    """
    Fetch a single training image patch from Sentinel-2
    
    Args:
        lat, lon: Coordinates
        label: 'mining' or 'natural'
        size: Image size in pixels
    
    Returns:
        NumPy array with shape (224, 224, 6) for 6 spectral bands
    """
    
    try:
        # Define area of interest
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(1000)  # 1km radius
        
        # Load Sentinel-2 data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(aoi) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # Get median composite
        image = sentinel.median().clip(aoi)
        
        # Select bands: Blue, Green, Red, NIR, SWIR1, SWIR2
        bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        
        # Export as NumPy array (you'll need to implement actual export)
        # This is a simplified version - real implementation would use ee.batch.Export
        
        print(f"✅ Fetched patch for {label} at ({lat}, {lon})")
        
        # For now, return mock data
        # Replace this with actual GEE export logic
        return np.random.rand(size, size, 6).astype(np.float32)
        
    except Exception as e:
        print(f"❌ Error fetching patch: {e}")
        return None

def create_training_dataset():
    """
    Create training dataset from known mining and natural locations
    
    You need to provide coordinates for:
    - Confirmed mining sites (positive examples)
    - Natural/protected areas (negative examples)
    """
    
    # Example coordinates - REPLACE WITH YOUR ACTUAL LABELED DATA
    mining_sites = [
        (33.7490, -115.4794),  # Eagle Mountain Mine
        (37.7749, -117.6321),  # Silver Peak
        # Add more known mining locations
    ]
    
    natural_sites = [
        (36.0544, -112.1401),  # Grand Canyon
        (37.8651, -119.5383),  # Yosemite
        # Add more natural/protected areas
    ]
    
    X_train = []
    y_train = []
    
    print("📊 Generating training dataset...")
    
    # Fetch mining patches (label = 1)
    for lat, lon in mining_sites:
        patch = fetch_training_patch(lat, lon, 'mining')
        if patch is not None:
            X_train.append(patch)
            y_train.append(1)
    
    # Fetch natural patches (label = 0)
    for lat, lon in natural_sites:
        patch = fetch_training_patch(lat, lon, 'natural')
        if patch is not None:
            X_train.append(patch)
            y_train.append(0)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"✅ Dataset created: {len(X_train)} samples")
    print(f"   - Mining samples: {np.sum(y_train == 1)}")
    print(f"   - Natural samples: {np.sum(y_train == 0)}")
    
    return X_train, y_train

# ============================================================================
# CNN MODEL ARCHITECTURE (Transfer Learning)
# ============================================================================

def build_model(input_shape=(224, 224, 6), num_classes=2):
    """
    Build CNN model using transfer learning
    
    Strategy: Use ResNet50 pre-trained on ImageNet as feature extractor,
    then add custom layers for satellite imagery classification
    """
    
    # NOTE: ResNet50 expects 3 channels (RGB), but we have 6 spectral bands
    # Solution: Use a 1x1 conv layer to reduce 6 channels to 3
    
    inputs = keras.Input(shape=input_shape)
    
    # Reduce 6 spectral bands to 3 channels using 1x1 convolution
    x = layers.Conv2D(3, (1, 1), padding='same', activation='relu')(inputs)
    
    # Load pre-trained ResNet50 (without top classification layer)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Freeze base model layers initially (transfer learning)
    base_model.trainable = False
    
    # Pass through ResNet50
    x = base_model(x, training=False)
    
    # Add custom classification layers
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='SpectraMining_CNN')
    
    return model, base_model

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the CNN model with transfer learning
    
    Two-stage training:
    1. Train only custom layers (base frozen)
    2. Fine-tune entire model (base unfrozen)
    """
    
    print("\n🚀 Building model architecture...")
    model, base_model = build_model()
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=2)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print("\n📊 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # ========================================================================
    # STAGE 1: Train with frozen base (Transfer Learning)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STAGE 1: Training custom layers (base model frozen)")
    print("="*80)
    
    history_stage1 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS // 2,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========================================================================
    # STAGE 2: Fine-tune entire model (Unfreeze base)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning entire model (base model unfrozen)")
    print("="*80)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    history_stage2 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS // 2,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history_stage1, history_stage2

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=2)
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Get predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Natural', 'Mining']))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
    
    if accuracy >= 0.92:
        print("✅ Target accuracy (92%) achieved!")
    else:
        print(f"⚠️ Below target accuracy. Need {(0.92-accuracy)*100:.2f}% improvement")
    
    return accuracy

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*80)
    print("SpectraMining CNN Training Pipeline")
    print("="*80)
    
    # Initialize GEE
    if not initialize_gee():
        print("\n⚠️ Running in MOCK mode - replace with real data for production")
    
    # Create or load dataset
    print("\n📥 Loading training data...")
    
    # Option 1: Load from saved files (if you have pre-processed data)
    if os.path.exists(DATASET_PATH):
        print(f"Loading from {DATASET_PATH}...")
        # Implement loading logic here
        X_train = np.load(os.path.join(DATASET_PATH, "X_train.npy"))
        y_train = np.load(os.path.join(DATASET_PATH, "y_train.npy"))
    else:
        # Option 2: Generate fresh dataset from GEE
        print("Generating fresh dataset from Google Earth Engine...")
        X_data, y_data = create_training_dataset()
        
        # Mock data for demonstration (REPLACE WITH REAL DATA)
        print("\n⚠️ Using MOCK data for demonstration")
        X_data = np.random.rand(100, 224, 224, 6).astype(np.float32)
        y_data = np.random.randint(0, 2, 100)
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Normalize data (0-1 range)
    X_train = X_train / 10000.0  # Sentinel-2 reflectance values
    X_val = X_val / 10000.0
    X_test = X_test / 10000.0
    
    # Train model
    model, hist1, hist2 = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save final model
    print(f"\n💾 Saving final model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Final test accuracy: {accuracy*100:.2f}%")
    print("="*80)
    
    # Instructions for using the model
    print("\n📝 Next Steps:")
    print("1. Replace mock AI classifier in app.py with this trained model")
    print("2. Load model using: model = keras.models.load_model('spectramining_model.h5')")
    print("3. Use model.predict() for real-time classification")

if __name__ == "__main__":
    main()
