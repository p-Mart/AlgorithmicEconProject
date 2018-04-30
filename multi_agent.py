import tensorflow as tf
import numpy as np
import sys, os, errno
import random
from tqdm import trange
import time

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def relu(x):
    z = np.zeros_like(x)
    return np.max(z, x)

def explore(x):
    x[x < 0] = 1. / np.prod(x.shape)
    return x

def onehot(x, size):
    y = np.zeros((size,))
    y[x] = 1.
    return y

def softmax(x):
    z = x - np.max(x)
    return np.exp(z) / np.sum(np.exp(z))

def replicator(G, x, y, delta=1.0):
    x, y = (softmax(x), softmax(y))

    A, B = G()

    delta_x = x*(A.dot(y) - x.T.dot(A).dot(y))
    delta_y = y*(x.T.dot(B).T - x.T.dot(B).dot(y))

    return delta_x, delta_y

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), (size, 5))

class DQN:
    def __init__(self, learning_rate, nb_actions, observation_shape, scope, h_size=128):
        with tf.variable_scope(scope):
            self.input = tf.placeholder(shape=[None]+list(observation_shape), dtype=tf.float32)

            # Architecture
            self.h_1 = tf.layers.dense(
                inputs=self.input,
                units=h_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1)
            )

            self.h_2 = tf.layers.dense(
                inputs=self.h_1,
                units=h_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1)
            )

            self.flatten = tf.layers.flatten(
                inputs = self.h_2
            )

            self.policy = tf.layers.dense(
                inputs=self.flatten,
                units=nb_actions,
                activation=tf.nn.softmax,
                use_bias=False,
                name="policy_out"
            )

            self.output =tf.layers.dense(
                inputs=self.flatten,
                units=nb_actions
            )

            self.predict = tf.argmax(self.output, axis=1)

            # Everything needed to calculate loss function
            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32)

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions_onehot), axis=1)

            self.loss = tf.losses.mean_squared_error(self.target_q, self.q)
            self.policy_loss = tf.reduce_sum(tf.multiply(-tf.log(self.policy), self.actions_onehot), axis=1)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.update_q = self.optimizer.minimize(self.loss)

            # Gradient blocking on policy branch
            with tf.variable_scope('policy_out', reuse=True):
                w = tf.get_variable('kernel')

            self.update_policy = self.optimizer.minimize(self.policy_loss, var_list=[w])

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def updateTargetNetwork(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def Battle():
    A = [
        [1, -1],
        [-1, 1]
    ]

    B = [
        [-1, 1],
        [1, -1]
    ]

    return np.array(A), np.array(B)

def summaryWrite(name, value, summary_writer, step):
    new_summary = tf.Summary()
    new_summary_value = new_summary.value.add()
    new_summary_value.simple_value = value
    new_summary_value.tag = name

    summary_writer.add_summary(new_summary, (step + 1))
    summary_writer.flush()

def main(model_name, options):
    # Hyperparameters
    batch_size = 32
    update_freq = 30
    save_freq = 200
    target_model_update = 0.001
    start_eps = 1.0
    end_eps = 0.1
    learning_rate = 0.003
    discount = 0.99
    replay_size = 50000
    nb_epochs = 10000
    nb_episodes = 100
    nb_episodes_test = 100
    h_size = 90
    env_name = 'simple_push.py'
    gamma = 0.4

    hyperparameters = {
        'batch_size' : batch_size,
        'update_freq' : update_freq,
        'save_freq' : save_freq,
        'target_model_update' : target_model_update,
        'start_eps' : start_eps,
        'end_eps' : end_eps,
        'learning_rate' : learning_rate,
        'discount' : discount,
        'replay_size': replay_size,
        'nb_episodes' : nb_episodes,
        'nb_episodes_test' : nb_episodes_test,
        'h_size' : h_size,
        'env_name' : env_name
    }

    # Initialize game
    scenario = scenarios.load(env_name).Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=False)

    # Env parameters
    nb_agents = env.n
    observation_shape = env.observation_space
    action_shape = env.action_space

    # Initialize networks, tensorflow vars
    tf.reset_default_graph()

    # DQN for player 1
    model_1 = DQN(
        learning_rate=learning_rate,
        h_size=32,
        nb_actions=action_shape[0].n,
        observation_shape=observation_shape[0].shape,
        scope='model_1'
    )

    target_model_1 = DQN(
        learning_rate=learning_rate,
        h_size=32,
        nb_actions=action_shape[0].n,
        observation_shape=observation_shape[0].shape,
        scope='target_model_1'
    )

    # DQN for player 2
    model_2 = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=action_shape[1].n,
        observation_shape=observation_shape[1].shape,
        scope='model_2'
    )

    target_model_2 = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=action_shape[1].n,
        observation_shape=observation_shape[1].shape,
        scope='target_model_2'
    )

    # Tensorflow Stuff
    sess = tf.Session()

    summary_writer = tf.summary.FileWriter(logdir="./logs/{}".format(model_name),graph=sess.graph)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()

    target_ops_1 = update_target_graph('model_1', 'target_model_1')
    target_ops_2 = update_target_graph('model_2', 'target_model_2')

    experience_1 = ReplayBuffer(buffer_size=replay_size)
    experience_2 = ReplayBuffer(buffer_size=replay_size)

    sess.run(init)

    # Meta Strategy Initialization
    dummy = np.array([0, 1])

    meta_1 = np.array([0.2, 0.8])
    meta_2 = np.array([0.2, 0.8])

    p1_action_store = None
    p2_action_store = None

    p1_accumulated_reward = np.array([0.0, 0.0])
    p2_accumulated_reward = np.array([0.0, 0.0])

    total_steps = 0

    log_dir = "./data/{}/".format(model_name)
    # Weights will be loaded if weight file exists.
    if os.path.exists(log_dir) and "load" in options:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

    # Attempt to create log directory for this model.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Training loop, if enabled
    if "train" in options:
        episode = 0

        #Initial observation
        epoch = 0
        try:
            for epoch in trange(nb_epochs):

                # For logging rewards
                r_all1 = 0.
                r_all2 = 0.
                observation = env.reset()
                # Accumulate Playing
                for episode in trange(nb_episodes):
                    # Sample from meta distributions
                    idx1 = np.random.choice(dummy, p=meta_1)
                    idx2 = np.random.choice(dummy, p=meta_2)

                    p1_action, p2_action = None, None
                    # Player 1 takes an action
                    if idx1 == 0:
                        p1_action = np.random.randint(action_shape[0].n)
                    else:
                        p1_action = sess.run(model_1.predict, feed_dict={model_1.input:[observation[0]]})[0]

                    # Player 2 takes action
                    if idx2 == 0:
                        p2_action = np.random.randint(action_shape[1].n)
                    else:
                        p2_action = sess.run(model_2.predict, feed_dict={model_2.input: [observation[1]]})[0]


                    # Get the policies of each player(?)
                    #p1_policy = sess.run(model_1.policy,feed_dict={model_1.input:[observation]})
                    #p2_policy = sess.run(model_2.policy,feed_dict={model_2.input:[observation]})


                    # Take collective action in environment
                    action_collective = [onehot(p1_action, action_shape[0].n),
                                         onehot(p2_action, action_shape[1].n)]

                    observation_next, reward_collective, done, _ = env.step(action_collective)
                    p1_reward, p2_reward = reward_collective

                    #reward_random = np.array([[0.5, 0.5]]).dot(A).dot(p2_policy.T)[0,0]
                    #reward = p1_policy.dot(A).dot(p2_policy.T)[0,0]

                    r_all1 += p1_reward
                    r_all2 += p2_reward

                    p1_accumulated_reward[idx1] += p1_reward
                    p2_accumulated_reward[idx2] += p2_reward

                    #reward = A[p1_action][p2_action]
                    #done = True if (episode == nb_episodes - 1) else False

                    # Store into experience buffer
                    new_experience1 = np.array([observation[0], p1_action, p1_reward, observation_next[0], done[0]])
                    new_experience1 = new_experience1.reshape((1, 5))
                    experience_1.add(new_experience1)

                    new_experience2 = np.array([observation[1], p2_action, p2_reward, observation_next[1], done[1]])
                    new_experience2 = new_experience2.reshape((1, 5))
                    experience_2.add(new_experience2)

                    observation = observation_next

                # Train Player 1
                train_batch = experience_1.sample(batch_size)

                s0_batch = np.stack(train_batch[:, 0], axis=0)
                s1_batch = np.stack(train_batch[:, 3], axis=0)

                model_q = sess.run(model_1.predict, feed_dict={model_1.input: s1_batch})
                target_model_q = sess.run(target_model_1.output, feed_dict={target_model_1.input: s1_batch})

                end_multiplier = 1 - train_batch[:, 4]
                double_q = target_model_q[range(batch_size), model_q]
                target_q = train_batch[:, 2] + (discount * double_q * end_multiplier)

                # Update model
                _ = sess.run([model_1.update_q, model_1.update_policy], feed_dict={
                    model_1.input: np.stack(s0_batch, axis=0),
                    model_1.target_q: target_q,
                    model_1.actions: train_batch[:, 1]
                })

                # Update meta strategy (Hedge Matching without exploration)
                #meta_1 = softmax((gamma / 3) * p1_accumulated_reward)

                # Update target network 1
                updateTargetNetwork(target_ops_1, sess)

                # Train Player 2
                train_batch = experience_2.sample(batch_size)

                s0_batch = np.stack(train_batch[:, 0], axis=0)
                s1_batch = np.stack(train_batch[:, 3], axis=0)

                model_q = sess.run(model_2.predict, feed_dict={model_2.input: s1_batch})
                target_model_q = sess.run(target_model_2.output, feed_dict={target_model_2.input: s1_batch})

                end_multiplier = 1 - train_batch[:, 4]
                double_q = target_model_q[range(batch_size), model_q]
                target_q = train_batch[:, 2] + (discount * double_q * end_multiplier)

                # Update model
                _ = sess.run([model_2.update_q, model_2.update_policy], feed_dict={
                    model_2.input: np.stack(s0_batch, axis=0),
                    model_2.target_q: target_q,
                    model_2.actions: train_batch[:, 1]
                })

                # Update meta strategy (Hedge Matching without exploration)
                #meta_2 = softmax((gamma / 3) * p2_accumulated_reward)

                # Update target network
                updateTargetNetwork(target_ops_2, sess)

                # Log reward in tensorboard
                #summaryWrite('player1/p1_action', p1_action_store, summary_writer, epoch+1)
                summaryWrite('player1/meta_1', meta_1[0], summary_writer, epoch+1)
                summaryWrite('player1/meta_2', meta_1[1], summary_writer, epoch+1)
                summaryWrite('player1/reward', r_all1, summary_writer, epoch+1)
                #summaryWrite('player1/policy_1', p1_policy[0,0], summary_writer, epoch+1)
                #summaryWrite('player1/policy_2', p1_policy[0,1], summary_writer, epoch+1)

                #summaryWrite('player2/p2_action', p2_action_store, summary_writer, epoch+1)
                summaryWrite('player2/meta_1', meta_2[0], summary_writer, epoch+1)
                summaryWrite('player2/meta_2', meta_2[1], summary_writer, epoch+1)
                summaryWrite('player2/reward', r_all2, summary_writer, epoch+1)

                #summaryWrite('player2/policy_1', p2_policy[0,0], summary_writer, epoch+1)
                #summaryWrite('player2/policy_2', p2_policy[0,1], summary_writer, epoch+1)

                # Save model occasionally
                if epoch % save_freq == 0:
                    p1_accumulated_reward = p1_accumulated_reward - np.max(p1_accumulated_reward)
                    p2_accumulated_reward = p2_accumulated_reward - np.max(p2_accumulated_reward)
                    saver.save(sess, log_dir+'/model-'+str(epoch+1)+'.ckpt')

        except KeyboardInterrupt:
            print("Training interrupted manually.")

        # Save model
        saver.save(sess, log_dir + '/model-' + str(epoch+1) + '.ckpt')

        # Log all hyperparameter settings
        #logHyperparameters(model_name, **hyperparameters)

    # Test loop, if enabled

    if "test" in options:
        try:
            for epoch in trange(nb_episodes_test):
                observation = env.reset()
                #r_all = 0
                #steps = 0

                for episode in trange(50):

                    # Only the network is used to take an action
                    action1 = sess.run(model_1.predict, feed_dict={model_1.input:[observation[0]]})[0]
                    action2 = sess.run(model_2.predict, feed_dict={model_2.input:[observation[1]]})[0]

                    action_collective = [onehot(action1, action_shape[0].n),
                                         onehot(action2, action_shape[1].n)]

                    observation, reward_collective, done, _ = env.step(action_collective)

                    # Step through environment
                    env.render()



        except KeyboardInterrupt:
            print("Testing interrupted manually.")


    return


if __name__ == "__main__":

    model_name = None
    options = []

    if len(sys.argv) >= 3:
        model_name = sys.argv[1]

        mode = sys.argv[2]

        if mode != "train" and mode != "test":
            print("Usage: python main.py [model_name].h5 [train | test]")
            sys.exit()

        options = sys.argv[2:]

    else:
        print("Incorrect number of arguments.")
        print("Usage: python main.py [model_name].h5 [train | test]")
        sys.exit()

    main(model_name, options)