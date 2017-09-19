import numpy as np

from agent import Agent
from ttt_env import TicTacToeEnvironment

ACTIONS = 9
EPISODES = 100000
ALPHA = .1
GAMMA = .5
D = []


def q_learning_play(agent1, agent2):
    env = TicTacToeEnvironment()
    epsilon = 1.
    epsilon_decay = .95 / EPISODES

    for i in range(EPISODES):
        s = env.reset()
        terminated = False
        while not terminated:
            if np.random.rand() <= epsilon:
                a = env.step_sample()
            else:
                if env.is_circles_turn():
                    a = agent1.get_best_action(s, env)
                else:
                    a = agent2.get_best_action(s, env)

            s_prime, r, terminated = env.step(a)
            if env.is_circles_turn():  # the state changes after a step
                agent2.memorize_transition(s, a, s_prime)
            else:
                agent1.memorize_transition(s, a, s_prime)
            s = s_prime

        agent2.update_policy(r, GAMMA, ALPHA)
        if r == 5:
            r = -5
        elif r == -5:
            r = 5
        agent1.update_policy(r, GAMMA, ALPHA)
        epsilon -= epsilon_decay

        if i % (EPISODES // 100) == 0:
            print(f'Episode: {i}/{EPISODES}, epsilon: {epsilon}')

    agent1.save_q_table()
    agent2.save_q_table()


def interactive_play(agent1, agent2):
    env = TicTacToeEnvironment(random_start=True)

    for i in range(EPISODES):
        agent = agent1 if np.random.randint(2) else agent2
        agent_is_circles = False
        if agent == agent1:
            print("The agent shall begin.")
            agent_is_circles = True

        print('*' * 10, "Game", i + 1, '*' * 10)
        s = env.reset()
        env.render()
        terminated = False
        step = 1
        while not terminated:
            if env.is_circles_turn() and not agent_is_circles \
                    or not env.is_circles_turn() and agent_is_circles:
                a = int(input("Choose your action (0-8): "))
            else:
                print("Thinking...")
                if step == 1:
                    a = agent.get_best_action(s, env) \
                        if np.random.randint(2) \
                        else np.random.randint(9)
                else:
                    a = agent.get_best_action(s, env)

            s_prime, r, terminated = env.step(a)
            env.render()
            if not terminated:
                s = s_prime
            step += 1

        if agent_is_circles:
            if abs(r) == 5:
                r = -r

        if r == 5:
            print("You lost.")
        elif r == -5:
            print("You won.")
        elif r == -1:
            print("Draw.")


if __name__ == '__main__':
    agent1 = Agent('circle_agent')
    agent2 = Agent('cross_agent')

    if not agent1.loaded or not agent2.loaded:
        print("Training the model...")
        q_learning_play(agent1, agent2)

    print("Now you go!")
    interactive_play(agent1, agent2)
