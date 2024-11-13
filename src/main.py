import cv2, time
from ai2thor.controller import Controller

from Perception import Perception
from RPA import RPA

class SimuICLA:

    def __init__(self, scene: str, DIME: tuple, perception: Perception) -> None:
        
        self.grid_size : float = 0.25
        self.rotation_increment : int = 30

        self.controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene=scene,
            visibilityDistance=1.5,
            gridSize=self.grid_size,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=DIME[1],
            height=DIME[0],
            fieldOfView=60
        )

        self.perception = perception
        self.RPA = RPA(self)

    def setScene(self, scene: str) -> None:
        self.controller.reset(scene=scene)

    def loop(self) -> None:
        time.sleep(10)
        self.controller.step("Done")


    def showBoxedAgentView(self) -> None:

        img = self.controller.last_event.frame

        cv2.imwrite("input.jpg", img)

        img, out = self.perception.getResultsFromFF(img)

        out = self.perception.getClassFromFF(
            out,
            img.shape[:2]
        )

        img_boxed = self.perception.addBoxesToImg(
            img,
            *out
        )

        cv2.imwrite("test.jpg", img_boxed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    
if __name__ == "__main__":
    
    DIME = (800, 1200)

    controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene="FloorPlan1",
            visibilityDistance=1.5,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=DIME[1],
            height=DIME[0],
            fieldOfView=60
        )
    
    controller.reset()
    
    controller.step(
    action="RotateAgent",
    degrees=-90,
    returnToStart=True,
    speed=1,
    fixedDeltaTime=0.02
    )

    while True:
        controller.step(
        action="MoveArm",
        position=dict(x=0, y=1.5, z=0),
        coordinateSpace="armBase",
        restrictMovement=False,
        speed=1,
        returnToStart=True,
        fixedDeltaTime=0.02
        )

        time.sleep(2)
        
        controller.step(
        action="MoveArm",
        position=dict(x=0, y=0, z=0),
        coordinateSpace="armBase",
        restrictMovement=False,
        speed=1,
        returnToStart=True,
        fixedDeltaTime=0.02
        )

        time.sleep(2)

    # perc = Perception(
    #     (
    #         "models/YOLO/coco.names",
    #         "models/YOLO/yolov3.cfg",
    #         "models/YOLO/yolov3.weights"
    #     )
    # )

    # sim = SimuICLA("FloorPlan1", DIME, perc)

    # sim.showBoxedAgentView()

    # sim.loop()