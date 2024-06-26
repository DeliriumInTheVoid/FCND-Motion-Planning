class PlanningConst:
    COLLIDERS_FILE = 'colliders.csv'

    ALTITUDE = 23  # 55  #
    SAFETY_DISTANCE = 5

    GOAL_GLOBAL_POSITION = [-122.394280, 37.797760, ALTITUDE]  # small house after park (alt = 20)

    PRE_PLANING = True
    USE_HORIZON_PLANNER = True
    RUN_DRONE = True

    HORIZON_SIZE = 200
    HORIZON_RANDOM_SAMPLES = 50
    HORIZON_HEIGHT = 40
    CONNECT_NEAREST_SAMPLES = 10

    DRAW_PATH_PRUNE_TRACING = False
    DRAW_GLOBAL_PATH = True
    DRAW_HORIZON_STEPS = True
