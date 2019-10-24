# gym_slither

Must have a docker container running the quay.io/openai/universe.flashgames:latest image connected -p 10000:5900 and -p 10001:15900 for reward and other connections.

Uses neat-python to run genetic algorithm from gym.action_space actions and pixels from Slither.io. 

Works best with the "internet.SlitherIOEasy-v0" env from gym as it colors all food and opponents common colors.
