from locust import HttpUser, task, between
import random
import os

class PlantDiseaseUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Load test images"""
        # Adjust path to point to validation set
        self.test_images_dir = "data/test"
        self.test_images = []
        
        # Collect some test images
        if os.path.exists(self.test_images_dir):
            for root, dirs, files in os.walk(self.test_images_dir):
                for file in files[:50]:  # Take first 50 images found
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.test_images.append(os.path.join(root, file))
        else:
            print(f"Warning: Test images directory not found at {self.test_images_dir}")
    
    @task(10)  # Weight 10 - most common operation
    def predict_single(self):
        """Single image prediction"""
        if self.test_images:
            image_path = random.choice(self.test_images)
            try:
                with open(image_path, 'rb') as f:
                    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                    self.client.post("/predict", files=files)
            except Exception as e:
                print(f"Error opening file {image_path}: {e}")
    
    @task(2)  # Weight 2 - less frequent
    def predict_batch(self):
        """Batch prediction (5 images)"""
        if len(self.test_images) >= 5:
            selected = random.sample(self.test_images, 5)
            files = []
            try:
                # Open all files
                file_handles = []
                for img in selected:
                    f = open(img, 'rb')
                    file_handles.append(f)
                    files.append(('files', (os.path.basename(img), f, 'image/jpeg')))
                
                self.client.post("/predict-batch", files=files)
                
                # Close files
                for f in file_handles:
                    f.close()
            except Exception as e:
                print(f"Error in batch prediction: {e}")
    
    @task(3)  # Weight 3 - monitoring checks
    def check_monitoring(self):
        """Check monitoring endpoint"""
        self.client.get("/monitoring")
    
    @task(1)  # Weight 1 - health checks
    def health_check(self):
        """Health check"""
        self.client.get("/health")
