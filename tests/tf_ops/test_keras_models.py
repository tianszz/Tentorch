from src.tf_ops.models.keras_models import SimpleDenseClassifier
import tensorflow as tf
import numpy as np

class TestSimpleDenseClassifier(tf.test.TestCase):
    def setUp(self):
        self.model_params = {
            "layers": [16,8],
            "activation": 'relu',
            "batch_norm": False,
            "layer_norm": False,
            "target_dim": 1,
            "out_activation": 'softmax'
        }

        class_0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 50)
        class_1 = np.random.multivariate_normal([5, 5], [[1, 0.5], [0.5, 1]], 50)
        X = np.vstack((class_0, class_1))
        y = np.hstack((np.zeros(50), np.ones(50)))

        # Shuffle and split the data into training and testing sets
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]
        split_index = int(0.8 * X.shape[0])
        self.X_train, self.X_test = X[:split_index], X[split_index:]
        self.y_train, self.y_test = y[:split_index], y[split_index:]



    def test_simple_dense_classifier(self):

        model = SimpleDenseClassifier(self.model_params)
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        # Train the model
        history = model.fit(self.X_train, self.y_train, epochs=5, batch_size=32, validation_split=0.2)
        model.summary()

        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test)

        print(f"Test accuracy: {test_acc:.2f}")
