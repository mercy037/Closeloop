import torch
import numpy as np
from online_learning import ExpWeights

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def evaluate_agent_bandits(envs, agent, episode_max_steps, bandit_loss, greedy_bandit, n_episodes=16, n_arms=2, lr=0.90,
                           decay=0.90, epsilon=0.0, bandit_step=1):
    agent.eval()
    task_idx = agent.get_task()
    n_tasks = len(envs)

    # Bandit debug
    feedback, arm = np.empty((n_tasks, n_arms, n_episodes, episode_max_steps+1)), np.empty((n_tasks, n_episodes, episode_max_steps+1))
    mses = np.empty((n_tasks, n_arms, n_episodes, episode_max_steps + 1))
    feedback[:], arm[:], mses[:] = np.nan, np.nan, 0

    # TB
    bandit_probs, bandit_p = np.empty((n_tasks, n_arms, n_episodes)), np.empty((n_tasks, n_arms, n_episodes, episode_max_steps+1))
    bandit_probs[:], bandit_p[:] = np.nan, np.nan
    episodes_return_mean, corrects = {i: 0 for i in range(n_tasks)}, {i: [] for i in range(n_tasks)}
    episode_return = {i: 0 for i in range(n_tasks)}

    # iterate through envs / Tasks
    for i in range(n_tasks):
        env = envs[i]
        correct, iter = 0, 0
        logs_episodes_return_mean, logs_episodes_return, logs_episodes_num_frames = 0, 0, 0

        for j in range(n_episodes):
            state = env.reset()
            done = False
            logs_episodes_num_frames += 1
            logs_episode_return, logs_episode_num_frames, iter_episode = 0, 0, 0
            bandit = ExpWeights(arms=list(range(n_arms)), lr=lr, decay=decay, greedy=greedy_bandit, epsilon=epsilon)

            while not done:
                if iter_episode % bandit_step == 0:
                    idx = bandit.sample()
                arm[i, j, iter_episode] = idx
                bandit_p[i, :, j, iter_episode] = bandit.p
                iter += 1
                correct += 1 if idx == i else 0
                agent.set_task(idx, q_reg=False)
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                logs_episode_num_frames += 1

                if logs_episode_num_frames >= episode_max_steps:
                    done = True

                # get feedback for each arm - because we can easily.
                # We are comparing the main Q val to a fixed Q target which is chosen by the bandit
                scores = []

                with torch.no_grad():
                    next_action, log_prob_, _, _, _ = agent.actor(next_state, evaluate=True)
                    target_Q1, target_Q2 = agent.critic_target(torch.FloatTensor(next_state).unsqueeze(0), next_action.view(-1,1))
                    target_Q = reward + agent.gamma * (torch.min(target_Q1, target_Q2) - agent.alpha * log_prob_)

                for k in range(n_arms):
                    agent.set_task(k, q_reg=False)
                    current_Q1, current_Q2 = agent.critic(torch.FloatTensor(state).unsqueeze(0), torch.FloatTensor(action).view(-1,1))
                    mus1_ = current_Q1.detach().cpu().numpy()
                    mus2_ = current_Q2.detach().cpu().numpy()
                    mse = np.sqrt(np.mean((mus1_ - target_Q.cpu().numpy()) ** 2)) + np.sqrt(np.mean((mus2_ - target_Q.cpu().numpy()) ** 2))
                    mses[i, k, j, iter_episode] += mse

                    if bandit_loss == 'mse':
                        scores.append(min(1/mse, 50))
                        feedback[i, k, j, iter_episode] = mse
                    else:
                        raise ValueError

                state = next_state
                logs_episode_return += reward
                bandit.update_dists(scores)

                if done:

                    if logs_episodes_num_frames == 0:
                        episode_return[i] = logs_episode_return / logs_episode_num_frames
                        logs_episodes_return += logs_episode_return / logs_episode_num_frames
                    logs_episodes_return_mean = logs_episodes_return / logs_episodes_num_frames

                iter_episode += 1

            corrects[i].append(correct / iter)
            for m in range(len(bandit.p)):
                bandit_probs[i, m, j] = bandit.p[m] # last probability from the bandit

        episodes_return_mean[i] = logs_episodes_return_mean

    # Reset network to original task, head
    agent.set_task(task_idx, q_reg=False)
    agent.train()

    return episodes_return_mean, episode_return, corrects, {'mses': feedback if bandit_loss == 'mse' else np.empty(feedback.shape),
                                            'bandit_prob': bandit_probs}

def evaluate_agent_oracle(envs, agent, episode_max_steps, n_episodes=16):
    agent.eval()
    task_idx = agent.get_task()
    n_tasks = len(envs)
    episodes_return_mean = {i: 0 for i in range(n_tasks)}
    episode_return = {i: 0 for i in range(n_tasks)}

    for i in range(n_tasks):
        env = envs[i]
        agent.set_task(i, q_reg=False)
        logs_episodes_return_mean, logs_episodes_return, logs_episodes_num_frames = 0, 0, 0

        for j in range(n_episodes):
            state = env.reset()
            done = False
            logs_episodes_num_frames += 1
            logs_episode_return, logs_episode_num_frames, iter_episode = 0, 0, 0

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                logs_episode_num_frames += 1

                if logs_episode_num_frames >= episode_max_steps:
                    done = True

                state = next_state
                logs_episode_return += reward

                if done:

                    if logs_episodes_num_frames == 1:
                        episode_return[i] = logs_episode_return / logs_episode_num_frames
                    logs_episodes_return += logs_episode_return / logs_episode_num_frames
                    logs_episodes_return_mean = logs_episodes_return / logs_episodes_num_frames

                iter_episode += 1

        episodes_return_mean[i] = logs_episodes_return_mean

    agent.set_task(task_idx, q_reg=False)
    agent.train()

    return episodes_return_mean, episode_return