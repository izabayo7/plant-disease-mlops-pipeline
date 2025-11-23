import tensorflow as tf
import os
import json
import shutil
from datetime import datetime
import numpy as np

def retrain_model(model_path, data_dir, epochs=5):
    """
    Fine-tunes the model on new data. Supports adding NEW classes by expanding the model architecture.
    """
    print(f"[{datetime.now()}] Starting retraining process...")
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Skipping.")
        return False, "No new data found"

    # Check if there are any images
    total_files = sum([len(files) for r, d, files in os.walk(data_dir)])
    if total_files == 0:
        print("No images found in new_data directory.")
        return False, "No images found"

    try:
        # Load class names
        class_names_path = os.path.join(os.path.dirname(model_path), '..', 'class_names.json')
        with open(class_names_path, 'r') as f:
            existing_classes = json.load(f)
        print(f"Existing classes: {len(existing_classes)}")
        
        # Load the original model
        print(f"Loading original model from {model_path}...")
        base_model = tf.keras.models.load_model(model_path)
        
        # Detect new classes in uploaded data
        new_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        print(f"New data classes: {new_classes}")
        
        # Combine with existing classes
        all_classes = list(existing_classes)
        new_class_added = False
        for cls in new_classes:
            if cls not in all_classes:
                all_classes.append(cls)
                new_class_added = True
                print(f"🆕 NEW CLASS DETECTED: {cls}")
        
        num_classes = len(all_classes)
        print(f"Total classes after merge: {num_classes}")
        
        # CRITICAL: Check if model output size matches required classes
        model_output_size = base_model.output_shape[-1]
        print(f"Model output size: {model_output_size}, Required classes: {num_classes}")
        
        # Check if we need to expand the model (either new classes OR mismatched output size)
        if new_class_added or model_output_size != num_classes:
            if model_output_size != num_classes:
                print(f"⚠️  Model output mismatch! Model has {model_output_size} outputs but needs {num_classes}")
            
            print("⚠️  Rebuilding output layer...")
            
            # Find the layer BEFORE the final classification layer
            # The model structure is: EfficientNet -> GlobalAvgPool -> Dense(256) -> Dropout -> Dense(N)
            # We want to extract features from BEFORE the final Dense layer
            
            # Get the output from the second-to-last layer (before final Dense)
            # This should be after Dropout
            penultimate_layer = base_model.layers[-2]  # This is the Dropout layer
            x = penultimate_layer.output
            
            # Add new output layer with expanded classes
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions_new')(x)
            
            # Create new model
            model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            
            print(f"✅ Model rebuilt: {model_output_size} → {num_classes} classes")
            new_class_added = True  # Force combined training
        else:
            print("✅ No changes needed. Using existing model architecture.")
            model = base_model
        
        # Determine data directories
        # To prevent catastrophic forgetting, we need to train on BOTH:
        # 1. Original training data (existing 38 classes)
        # 2. New uploaded data (e.g., 4 banana classes)
        
        if new_class_added:
            print("⚠️  New classes detected. Will train on ORIGINAL + NEW data to prevent forgetting.")
            epochs = 5  # DEMO MODE: Reduced from 10 to 5 for speed
            
            # Create a combined data directory
            import tempfile
            combined_data_dir = tempfile.mkdtemp(prefix='combined_data_')
            
            # Get original training data path
            original_train_dir = os.path.join(os.path.dirname(model_path), '..', '..', 'data', 'train')
            
            if os.path.exists(original_train_dir):
                print(f"Copying original training data from {original_train_dir}...")
                
            if os.path.exists(original_train_dir):
                print(f"Copying original training data from {original_train_dir}...")
                
                # DEMO OPTIMIZATION: Sample only a subset of original data for speed
                # This prevents total forgetting but trains much faster than using all 60k images
                SAMPLE_SIZE = 50
                print(f"⚡️ DEMO MODE: Sampling {SAMPLE_SIZE} images per original class for fast retraining...")
                
                import random
                symlinked_count = 0
                
                for class_name in existing_classes:
                    src_dir = os.path.join(original_train_dir, class_name)
                    
                    if os.path.exists(src_dir) and os.path.isdir(src_dir):
                        dst_dir = os.path.join(combined_data_dir, class_name)
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        # Get all images
                        try:
                            images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            
                            # Sample subset
                            if len(images) > SAMPLE_SIZE:
                                selected_images = random.sample(images, SAMPLE_SIZE)
                            else:
                                selected_images = images
                                
                            # Symlink selected images
                            for img in selected_images:
                                src_img = os.path.join(src_dir, img)
                                dst_img = os.path.join(dst_dir, img)
                                os.symlink(src_img, dst_img)
                                
                            symlinked_count += 1
                        except Exception as e:
                            print(f"Warning: Could not sample {class_name}: {e}")
                
                print(f"✅ Sampled {symlinked_count} existing classes")
                
                # Then, copy the new uploaded data (ALL of it)
                print(f"Adding new uploaded data (keeping ALL new images)...")
                copied_count = 0
                for class_name in new_classes:
                    src = os.path.join(data_dir, class_name)
                    dst = os.path.join(combined_data_dir, class_name)
                    if os.path.exists(src):
                        try:
                            shutil.copytree(src, dst)
                            copied_count += 1
                        except Exception as e:
                            print(f"Warning: Could not copy {class_name}: {e}")
                
                print(f"✅ Copied {copied_count} new classes from uploaded data")
                
                training_data_dir = combined_data_dir
                total_classes = symlinked_count + copied_count
                print(f"✅ Combined dataset ready: {symlinked_count} original + {copied_count} new = {total_classes} total classes")
            else:
                print(f"⚠️  Original training data not found at {original_train_dir}")
                print(f"⚠️  Training only on new data (may cause forgetting).")
                training_data_dir = data_dir
        else:
            print("✅ Retraining on existing classes only")
            training_data_dir = data_dir
            epochs = 5  # Fewer epochs for fine-tuning existing classes
        
        # Create data generator
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        print(f"Preparing data generators from {training_data_dir}...")
        
        # Create base generators
        base_train_gen = datagen.flow_from_directory(
            training_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        base_val_gen = datagen.flow_from_directory(
            training_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # No need for label remapping when using combined directory -
        # the generator already has all classes in the right order
        # RemappedSequence is only needed when training on subset of classes
        train_generator = base_train_gen
        validation_generator = base_val_gen
        
        # OPTIMIZATION: Freeze base layers for faster, stable training
        # We only want to train the classification head (Dense layers)
        # This prevents destroying the pretrained features and speeds up training
        print("🥶 Freezing base layers (EfficientNet) for transfer learning...")
        trainable_layers = 0
        for layer in model.layers:
            # Keep Dense layers and the new output layer trainable
            if isinstance(layer, tf.keras.layers.Dense):
                layer.trainable = True
                trainable_layers += 1
            else:
                layer.trainable = False
        
        print(f"   - {trainable_layers} trainable layers (Dense head)")
        print(f"   - {len(model.layers) - trainable_layers} frozen layers (Base)")
        
        # Compile model
        # increased LR to 1e-4 since we are training only the head (faster convergence)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Starting training for {epochs} epochs...")
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator
        )
        
        # Save updated class names
        if new_class_added:
            print(f"Updating class names: {class_names_path}")
            with open(class_names_path, 'w') as f:
                json.dump(all_classes, f, indent=2)
        
        # Save the retrained model
        print("Saving retrained model...")
        retrained_path = model_path.replace('.keras', '_retrained_latest.keras')
        model.save(retrained_path)
        
        print(f"✅ Retraining complete!")
        print(f"   - Classes: {num_classes} ({len(new_classes)} uploaded, {len(existing_classes)} original)")
        print(f"   - Retrained model: {retrained_path}")
        print(f"   - Original model: {model_path} (unchanged)")
        
        return True, history.history
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)
