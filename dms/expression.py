"""Optional explicit ONNX CNN contract; no bundled or guessed weights."""
import json
import cv2
import numpy as np


class ExpressionCNN:
    def __init__(self, model, labels):
        self.labels = json.loads(labels.read_text(encoding='utf-8'))
        if not isinstance(self.labels, list) or not self.labels or not all(isinstance(x, str) for x in self.labels):
            raise ValueError('Labels must be a nonempty JSON array of strings')
        self.net = cv2.dnn.readNetFromONNX(str(model))

    def predict(self, frame, points):
        h, w = frame.shape[:2]
        lo = np.maximum(points[:, :2].min(axis=0).astype(int), 0)
        hi = np.minimum(points[:, :2].max(axis=0).astype(int), (w, h))
        crop = frame[lo[1]:hi[1], lo[0]:hi[0]]
        if not crop.size:
            return 'Expression unavailable'
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self.net.setInput(cv2.dnn.blobFromImage(gray, 1 / 255., (48, 48)))
        logits = self.net.forward().reshape(-1)
        if len(logits) != len(self.labels) or not np.isfinite(logits).all():
            raise ValueError('CNN output must contain one finite logit per label')
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        i = int(probabilities.argmax())
        return f'{self.labels[i]} ({probabilities[i]:.0%}, uncalibrated CNN score)'
