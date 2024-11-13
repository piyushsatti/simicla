import os, cv2, numpy as np

class Perception:

    def __init__(self, paths: tuple, confidence: float = 0.5) -> None:

        labels_path, config_path, weights_path = paths

        if not os.path.exists(labels_path) or not os.path.exists(weights_path) or not os.path.exists(config_path) :
            raise Exception("Error in YOLO path, could not initialize *Perception*")

        self.CONF = confidence

        print("[INFO] Loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        print("[INFO] Completed loading YOLO from disk.")

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        print("[INFO] Loading classes from disk...")
        self.classes = open(labels_path).read().strip().split("\n")
        print("[INFO] Completed loading classes from disk.")

        print("[INFO] Loading COLORS from disk...")
        np.random.seed(42)
        self.COLO = np.random.randint(
            0, 
            255, 
            size=(len(self.classes), 3),
            dtype="uint8"
        )
        print("[INFO] Completed loading COLORS from disk...")

        print("[DONE] Perception Initialized with: ", "confidence=", confidence)

    def getResultsFromFF(self, img: np.ndarray) -> tuple:

        blob = cv2.dnn.blobFromImage(
            img, 
            1/255.0, 
            (416,416),
            swapRB=True, 
            crop=False
        )

        self.net.setInput(blob)
        
        outputs = np.vstack(
            self.net.forward(self.ln)
        )

        return img, outputs

    def getClassFromFF(self, outputs: np.vstack, img_dimensions: tuple) -> tuple:

        H, W = img_dimensions
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            
            scores = output[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            if conf <= self.CONF:
                continue

            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(conf))
            classIDs.append(classID)
        
        return (boxes, confidences, classIDs)

    def addBoxesToImg(self, img: np.ndarray, boxes: list, confidences: list, classIDs: list) -> np.ndarray:
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONF, self.CONF-0.1)

        if len(indices) <= 0:
            print("[INFO] No indices to add bounding boxes.")
            return
        
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in self.COLO[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return img
    

if __name__ == "__main__":

    perc = Perception(
        (
            "models/YOLO/coco.names",
            "models/YOLO/yolov3.cfg",
            "models/YOLO/yolov3.weights"
        )
    )

    img = cv2.imread("assets/images/horse.jpg")

    img, out = perc.getResultsFromFF(img)

    out = perc.getClassFromFF(
        out,
        img.shape[:2]
    )

    img_boxed = perc.addBoxesToImg(
        img,
        *out
    )

    cv2.imshow("test", img_boxed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()