from main import SimuICLA

class RPA:

    def __init__(self, SimuICLA: SimuICLA) -> None:
        self.SimuICLA = SimuICLA

        self.return_to_start = True
        self.speed = 1
        self.fixed_time_delta = 0.02 

    def moveAgent(self, ahead : int = 0, right : int = 0, left = 0):
        
        # Make sure non-negative
        if not (left >= 0 or right >= 0):
            raise Exception("[ERRO] moveAgent: left or right had negative values.")

        # Correction if both are non-zero
        if right != 0 and left > right:
            left = left - right
            right = 0
        if left !=0 and right > left:
            right  = right - left
            left = 0

        left = left * self.SimuICLA.grid_size
        right = right * self.SimuICLA.grid_size

        if left == 0:
            self.SimuICLA.controller.step(
                action="MoveAgent",
                ahead=ahead,
                right=right,
                returnToStart=self.return_to_start,
                speed=self.speed,
                fixedDeltaTime=self.fixed_time_delta
            )
        else:
            self.SimuICLA.controller.step(
                action="MoveAgent",
                ahead=ahead,
                left=left,
                returnToStart=self.return_to_start,
                speed=self.speed,
                fixedDeltaTime=self.fixed_time_delta
            )

    def rotateAgent(self, left: int = 0, right : int = 0):
        
        # Make sure non-negative
        if not (left >= 0 or right >= 0):
            raise Exception("[ERRO] moveAgent: left or right had negative values.")

        # Correction if both are non-zero
        angle = right - left

        angle = angle * self.SimuICLA.rotation_increment

        self.SimuICLA.controller.step(
            action="RotateAgent",
            degrees=angle,
            returnToStart=self.return_to_start,
            speed=self.speed,
            fixedDeltaTime=self.fixed_time_delta
        )
    
    def moveArm(self, target : dict):
        
        # target is the true x,y,z of where to move
        event = self.SimuICLA.controller.last_event
        event.metadata["agent"]["position"]

        self.SimuICLA.controller.step(
            action="MoveArm",
            position=dict(x=0, y=0.5, z=0),
            coordinateSpace="armBase",
            restrictMovement=False,
            returnToStart=self.return_to_start,
            speed=self.speed,
            fixedDeltaTime=self.fixed_time_delta
        )

if __name__ == "__main__":
    pass
