# 755_A1

Tested on windows 10 
Dependancies:
  Only dependancy needed that was not used in tutorials is Seaburn (Note Category_Encoders was causing my enviroment to crash with Seaburn)

Commands To Test (Note models are not saved on repo so models will be generated then tested, random_state is set at 42 for deterministic splitting)

World cup:
  python world_cup.py classification "path to test file"
  python world_cup.py regression "path to test file"
Traffic Volume
  python traffic_flow.py "path to test file"
Occupancy 
  python occupancy.py "path to test file"
Landsat
  python landsat.py "path to test file"
