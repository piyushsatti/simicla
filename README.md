Modules Needed
1. Make module for the object interaction layer
2. Make module called perception
3. Make module for the navigation layer

- Perception Layer (event["metadata"])
    - Input from `event`: RGB Image, Point Cloud
    - Output to CLM: Feature map
    - Output to RPA: 2D Bounding Boxes, 3D coordinates
- CLM Layer
    - Input from Perception Layer: Feature Map
    - Input from Interaction Layer: Semantic Label
    - Output to RPA: ???
- RPA
    - Semantic Reasoning
        - Input from Interaction Layer: Task Request
        - Input from CLM Layer: ???
    - Task Planner
        - Input from Perception Layer: 2D Bounding Boxes, 3D Coordinates
    - Action Sequences (`controller.step`)
        - Arm Control
        - Base Contorl

# To-Do
- [ ] Actor
    - [ ] RGB Camera (come pre built)
    - [ ] Room Observation Camera (can be added and recorded)
    - [ ] Lidar Sensor (depth map image available)
- [ ] Create Rooms
    - [ ] Kitchen
    - [ ] Home office
    - [ ] Dining Area
- [ ] Number of Objects
    - [ ] 20 Objects
    - [ ] 30 Objects
- [ ] Configure SLAM Mapping
    - [ ] Test Navigation Inside of an area
    - [ ] Test Navigation Between Areas
        - [ ] Kitchen  - Home Office
        - [ ] Kitcehn - Dining Area
        - [ ] Home Office - Dining Area
- [ ] Move Objects
    - [ ] Within the same area
    - [ ] Between areas
    

# Things learned
- Base YOLO as per docs does not work - would need to replace `i[0]` w/t `i` to resolve scalar issue.
- cv2 displaywindow methods are not functional on opencv-python install. Conda enviornments and imports work.  